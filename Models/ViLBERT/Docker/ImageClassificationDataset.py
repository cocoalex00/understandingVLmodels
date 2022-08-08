# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the dataset class necessary to fine-tune ViLBERT to an image classification task.
#   - The textual input to the model is "[CLS][SEP]", to denote that there is absence of it while making use of the special tokens used in its pre-training

from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import _pickle as cPickle
from torch.utils.data.distributed import DistributedSampler

#from pytorch_transformers.tokenization_bert import BertTokenizer

import jsonlines
import sys
import pdb

from ._image_features_reader import ImageFeaturesH5Reader

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




def _load_dataset(annotations_path):  
    """Load entries from a jsonline file

    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    annotations_pathI: path to the jsonline file for the dataset
    """
    with jsonlines.open(annotations_path) as reader:

        items = []
        count = 0
        for annotation in reader:
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


def get_data_loader(annotations_jsonpath: str, lmdb_path: str, batch_size: int = 16, max_region_num: int=37, shuffle: bool=False, multi_gpu: bool=False):
    """Get a data loader ready from a lmdb and jasonline files

    annotations_jsonpath: path to the jsonline file for the dataset
    lmdb_path: path to lmdb file containing the image features
    batch_size: number of samples in each batch 
    max_region_num: no idea
    shuffle: suffle the dataset or not
    """

    #Create the feature reader and dataset
    features_reader = ImageFeaturesH5Reader(lmdb_path)

    dataset = ImageClassificationDataset(annotations_jsonpath, features_reader,max_region_num)

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
        image_features_reader: ImageFeaturesH5Reader,
        max_region_num: int = 37
    ):
        super().__init__()
        self._max_region_num = max_region_num
        self.num_labels = 365
        self._image_features_reader = image_features_reader
        self.entries = _load_dataset(annotations_jsonpath)


    def __getitem__(self, index):

        
        entry = self.entries[index]                                             # Read input
        db_id = entry["db_id"]                                                  
        features, num_boxes, boxes, _ = self._image_features_reader[db_id]      # Read image features
        
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
        textInput = torch.tensor(entry["text_input"])


        co_attention_mask = torch.zeros((self._max_region_num, 30))
        target = torch.zeros(self.num_labels).float()                                   # One-hot vector of the entry's target

        labels = entry["label"]
        if labels is not None:
            target.scatter_(0, torch.tensor(labels), 1)

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