# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the dataset class necessary to fine-tune ViLBERT to an image classification task.
#   - The textual input to the model is "[CLS][SEP]", to denote that there is absence of it while making use of the special tokens used in its pre-training

import copy
from typing import Any, Dict, List
import random
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import _pickle as cPickle
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import random 
from transformers import BertTokenizer

#from pytorch_transformers.tokenization_bert import BertTokenizer


import json
import time
import csv
import base64

# long fields of the TSV file break python so we need to set the size limit higher up 
csv.field_size_limit(sys.maxsize)

############################ UTILS (Helper functions) #########################################

def _create_entry(item): 
    entry = {
        "id": item["id"],
        "img_path": item["img_path"],
        "label": item["label"],
        "db_id": item["db_id"],
        "text_input": item["text_input"],
        "input_mask": item["input_mask"],
        "segment_ids": item["segment_ids"],
    }
    return entry

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]



def _load_dataset(annotations_path):  
    """Load entries from a jsonline file

    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    annotations_pathI: path to the jsonline file for the dataset
    """
    with open(annotations_path) as reader:

        items = []
        count = 0
        for annotation in json.load(reader):
            dictionary = {}
            dictionary["id"] = int(annotation["id"])
            dictionary["img_path"] = str(annotation["img_path"])
            dictionary["label"] = int(annotation["label"])
            # "[CLS] [SEP]" == [101, 102] when encoded by bert tokenizer
            dictionary["text_input"] = [101,102]

            ### FIND OUT REAL NAME OF FILES FOR SERVER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            dictionary["db_id"] = str(annotation["img_name"])

            dictionary["input_mask"] = [1] * len(dictionary["text_input"])
            dictionary["segment_ids"] = [0] * len(dictionary["text_input"])
            items.append(dictionary)
            count += 1
            
    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries

 

 ############################ DATASET CLASS #######################################

class ITretrievalPlaces(Dataset):
    def __init__(
        self,
        annotations_jsonpath: str,
        labels_path: str,
        TSV_path,
        #image_features_reader: ImageFeaturesH5Reader,
        max_region_num: int = 37,
        test: bool=False,
        padding_index = 0
        
    ):
        super().__init__()
        self._max_region_num = max_region_num
        self.num_labels = 365
        self.entries = _load_dataset(annotations_jsonpath)
        self.tsvroot = TSV_path

        self.test = test
        self.labels_to_text = pd.read_csv(labels_path, header=None, delimiter = "/")[0].values.tolist()
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.tokenized_labels = []
        self._padding_index = padding_index
        self.tokenize()


    def tokenize(self, max_length=7):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.labels_to_text:
            tokens = self._tokenizer.tokenize(entry)
            tokens = tokens[: max_length - 2]
            tokens = [["CLS"]] + tokens + [["SEP"]]
            #print(tokens)

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]

            tokens = self._tokenizer.encode(entry)

            #tokens = tokens[: max_length - 2]
            

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            #assert(len(tokens), max_length)
            self.tokenized_labels.append([tokens,input_mask,segment_ids])



    def __getitem__(self, index):

        
        entry = self.entries[index]                                             # Read input
        db_id = entry["db_id"]  
        pathname = self.tsvroot + db_id + ".tsv"
        #print(self.image_features)
        # Read image features from tsv file                                                
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
                num_boxes = data["num_boxes"] 
                boxes = data["boxes"]      # Read image features
                g_feat = np.sum(features, axis=0) / num_boxes
                boxes = data["boxes"]      # Read image features
                image_w = data["img_w"]
                image_h = data["img_h"]
        
                features = np.concatenate(
                    [np.expand_dims(g_feat, axis=0), features], axis=0
                )

                num_boxes = num_boxes + 1

                image_location = np.zeros((boxes.shape[0],5), dtype=np.float32)
                image_location[:,:4] = boxes
                image_location[:,4] = (
                    (image_location[:,3] - image_location[:,1])
                    * (image_location[:,2] - image_location[:,0])
                    / (float(image_w) * float(image_h))
                )

                image_location_ori = copy.deepcopy(image_location)
                image_location[:,0] = image_location[:,0] / float(image_w)
                image_location[:,1] = image_location[:,1] / float(image_h)
                image_location[:,2] = image_location[:,2] / float(image_w)
                image_location[:,3] = image_location[:,3] / float(image_h)

                g_location = np.array([0,0,1,1,1])
                image_location = np.concatenate(
                    [np.expand_dims(g_location, axis=0), image_location], axis=0
                )

                boxes = image_location

        # Preprocess visual features with masks and padding and stuff
        mix_num_boxes = min(int(num_boxes), self._max_region_num)               # Things for image masks and stuff 
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()                        # Convert everything to tensors before sending it out
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        

        co_attention_mask = torch.zeros((self._max_region_num, 30))
        #print(co_attention_mask)
                                  # One-hot vector of the entry's target

        # if not self.test:
        #     labels = entry["label"]
        # else: 
        #     labels = None
        
        # if labels is not None:
        #     target.scatter_(0, torch.tensor(labels), 1)
        # else:
        #     target = 0

        # Text part 
        # Randomly select either the correct text or a random incorrect one
        if random.choice([0,1]) == 1:
            textInput = torch.tensor(self.tokenized_labels[entry["label"]][0])
            input_mask = torch.tensor(self.tokenized_labels[entry["label"]][1])
            segment_ids = torch.tensor(self.tokenized_labels[entry["label"]][2])
            target = 1
        else:
            randomID = random.choice(list(set(range(0, 364)) - set([entry["label"]])))
            textInput = torch.tensor(self.tokenized_labels[randomID][0])
            input_mask = torch.tensor(self.tokenized_labels[randomID][1])
            segment_ids = torch.tensor(self.tokenized_labels[randomID][2])
            target = 0

        return (
            textInput,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            #question_id,
            target,
        )

    def __len__(self):
        return len(self.entries)
