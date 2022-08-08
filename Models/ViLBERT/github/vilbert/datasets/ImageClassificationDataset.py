# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the dataset class necessary to fine-tune ViLBERT to an image classification task.
#   - The textual input to the model is "[CLS][SEP]", to denote that there is absence of it while making use of the special tokens used in its pre-training

from typing import Any, Dict, List
import random
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import _pickle as cPickle
from torch.utils.data.distributed import DistributedSampler

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
        "segment_ids": item["segment_ids"]
    }
    return entry

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def _load_obj_tsv(fname,start = 0, topk=None):
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
        if start != 0:
            for _ in range(start):
                next(f)
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
            dictionary["db_id"] = str(annotation["img_path"].split(".")[0])

            dictionary["input_mask"] = [1] * len(dictionary["text_input"])
            dictionary["segment_ids"] = [0] * len(dictionary["text_input"])
            items.append(dictionary)
            count += 1
            
    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


def get_data_loader(annotations_jsonpath: str, TSV_path: str, batch_size: int = 16, max_region_num: int=37, shuffle: bool=False, multi_gpu: bool=False, test: bool= False):
    """Get a data loader ready from a lmdb and jasonline files

    annotations_jsonpath: path to the jsonline file for the dataset
    TSV_path: path to TSV file containing the image features
    batch_size: number of samples in each batch 
    max_region_num: no idea
    shuffle: suffle the dataset or not
    """

    #Create the feature reader and dataset
    image_features = _load_obj_tsv(TSV_path)

    dataset = ImageClassificationDataset(annotations_jsonpath, image_features,max_region_num,test)

    if multi_gpu:
        sampler = DistributedSampler(dataset=dataset)

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )

    return dataloader
    

 

 ############################ DATASET CLASS #######################################

class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        annotations_jsonpath: str,
        image_features,
        #image_features_reader: ImageFeaturesH5Reader,
        max_region_num: int = 37,
        test: bool=False
    ):
        super().__init__()
        self._max_region_num = max_region_num
        self.num_labels = 365
        self.image_features = image_features
        self.entries = _load_dataset(annotations_jsonpath)
        self.imgid2img = {}
        for img_datum in image_features:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.test = test


    def __getitem__(self, index):

        
        entry = self.entries[index]                                             # Read input
        db_id = entry["db_id"]  
        #print(self.image_features)
        # Read image features from tsv file                                                
        features = self.imgid2img[db_id]["features"]
        num_boxes = self.imgid2img[db_id]["num_boxes"] 
        boxes = self.imgid2img[db_id]["boxes"]      # Read image features
        

        # Preprocess visual features with masks and padding and stuff
        mix_num_boxes = min(int(num_boxes), self._max_region_num)               # Things for image masks and stuff 
        mix_boxes_pad = np.zeros((self._max_region_num, 4))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()                        # Convert everything to tensors before sending it out
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        textInput = torch.tensor(entry["text_input"])

        co_attention_mask = torch.zeros((self._max_region_num, 30))
        target = torch.zeros(self.num_labels)                                  # One-hot vector of the entry's target

        if not self.test:
            labels = entry["label"]
        else: 
            labels = None
        
        if labels is not None:
            target.scatter_(0, torch.tensor(labels), 1)
        else:
            target = 0

        input_mask = torch.from_numpy(np.array(entry["input_mask"]))
        segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
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