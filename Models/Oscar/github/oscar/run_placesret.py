
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random, copy, time, json
import pandas as pd

import sys
sys.path.insert(0, '.')


import numpy as np
import torch
import torch.nn as nn
import csv
csv.field_size_limit(sys.maxsize)
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import _pickle as cPickle

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule
import base64

from oscar.modeling.modeling_bert import ImageBertForMultipleChoice, ImageBertForSequenceClassification

from torch.optim import Adamax
from oscar.utils.task_utils import (_truncate_seq_pair, output_modes, processors)

logger = logging.getLogger(__name__)

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

log_json = []

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data




def _load_dataset(args, name,val=False):
    processor = processors[args.task_name]()
    if name == 'train':
        examples = processor.get_train_examples(args.data_dir,'places365_train_alexsplit.json')
    elif name == 'val':
        examples = processor.get_dev_examples(args.data_dir, 'places365_retrieVal.json',val)
    elif name == 'test':
        examples = processor.get_dev_examples(args.data_dir, 'places365_test_small.json')
    return examples

class Places365(Dataset):
    """ Places365 Dataset """

    def __init__(self, args, name, img_features_path, tokenizer,labelsPath,val= False):
        super(Places365, self).__init__()
        assert name in ['train', 'val',"test"]

        # Convert to dictionary 
        self.img_features_path = img_features_path
        self.val = val

        self.tokenizer = tokenizer
        self.args = args
        self.name = name
        self.labels_to_text = pd.read_csv(labelsPath, header=None, delimiter = "/")[0].values.tolist()

        self.examples = _load_dataset(args, name,val)

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        # The negative and positive pairs are set for the validation 
        if self.val == True:
            text = self.labels_to_text[example.text_a]
            label_id = example.label
        else:
        # the positive and negative pairs are chosen randomly during training
            if random.choice([0,1]) == 1:
                text = self.labels_to_text[example.label]
                label_id = 1
            else:
                randomID = random.choice(list(set(range(0, 364)) - set([example.label])))
                text = self.labels_to_text[randomID]
                label_id = 0
    
        # tokenize sentence
        tokens= self.tokenizer.tokenize(text)   # Tokens are only [CLS] [SEP]

        # Check if larger than maxlength
        if len(tokens) > self.args.max_seq_length - 2:
                tokens = tokens[:(self.args.max_seq_length - 2)]

        # Add separator tokens to tokens and create the segment ids
        tokens = tokens + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # add class token to the tokens string and to the segment ids
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        # Convert the tokenized sentence to ids 
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)


        # No zero padding to the right 
        padding_length = self.args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length

        # image features

        img_key = example.img_key
        #print(img_key)
        pathname = self.img_features_path + img_key + ".tsv"
        with open(pathname) as f:
            for data in csv.DictReader(f, FIELDNAMES, delimiter="\t"):

                for key in ['img_h', 'img_w', 'num_boxes']:
                    data[key] = int(data[key])
            
                boxes = data['num_boxes']
                decode_config = [
                    ('objects_id', (boxes, ), np.int64),
                    ('objects_conf', (boxes, ), np.float32),
                    ('attrs_id', (boxes, ), np.int64),
                    ('attrs_conf', (boxes, ), np.float32),
                    ('boxes', (boxes, 4), np.float32),
                    ('features', (boxes, -1), np.float32),
                ]
                for key, shape, dtype in decode_config:
                    data[key] = np.frombuffer(base64.b64decode(data[key]), dtype=dtype)
                    data[key] = data[key].reshape(shape)
                    data[key].setflags(write=False)

                features = data["features"]


                # Last 6 elements of the visual features are calculated here
                boxcoordinates = data["boxes"]
                imgwh = np.array([data['img_w'],data['img_h'],data['img_w'],data['img_h']], dtype=np.float32)

                sapearray = boxcoordinates/imgwh

                cuidao = np.hstack(( sapearray,np.array([sapearray[:,3]-sapearray[:,1], sapearray[:,2]-sapearray[:,0]]).T))

                features = np.hstack((features,cuidao))

                img_feat = torch.tensor(features, dtype = torch.long)

                # print(data["boxes"])
                # print("------------------------------------")
                # print(data["boxes"].shape)
                # print(data["features"].shape)
                # print(data["boxes"][0])
                #num_boxes = data["num_boxes"] 
                #boxes = data["boxes"]      # Read image features

        #img_feat = torch.tensor(self.imgid2img[img_key]["features"], dtype= torch.long)
        if img_feat.shape[0] > 2*self.args.max_img_seq_length:
            img_feat = img_feat[0: 2*self.args.max_img_seq_length, ]
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((2*self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            print(img_feat.shape[1])
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(label_id, dtype=torch.long),
                img_feat)

    def __getitem__(self, index):
        entry = self.examples[index]

        example = self.tensorize_example(entry,
            cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
            pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example

    def __len__(self):
        return len(self.examples)

