# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
from torch.utils.data import Dataset
import torch

from .param import args
from .utils import load_obj_tsv
import base64
import csv
import sys 
import pandas as pd
import random 
csv.field_size_limit(sys.maxsize)

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


class PlacesDataset:
    """
    A Places data example in json file:
    {
        "id": 4,
        "img_name": "purrela-Places365_val_00000005",
        "img_path": "Places365_val_00000005.jpg",
        "label": 289
    }
    """
    def __init__(self, anotations_path: str, imgfeature_path: str, labels_path: str, val: bool = False):
        self.Apath = anotations_path
        self.imgfeatpath = imgfeature_path
        self.labels_to_text = pd.read_csv(labels_path, header=None, delimiter = "/")[0].values.tolist()
        self.val = val

        # Loading datasets to data
        self.data = []
        self.data.extend(json.load(open(self.Apath)))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['img_name']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class PlacesTorchDataset(Dataset):
    def __init__(self, dataset: PlacesDataset, test: bool=False):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        self.test = test

        # Loading detection features to img_data
        #img_data = []
        #img_data.extend(load_obj_tsv(self.raw_dataset.imgfeatpath, topk=topk))
        # self.imgid2img = {}
        # for img_datum in img_data:
        #     self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        self.data = self.raw_dataset.data
        # for datum in self.raw_dataset.data:
        #     if datum['img_name'] in self.imgid2img:
        #         self.data.append(datum)
        
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_name']
        
        if self.raw_dataset.val == False:
            if random.choice([0,1]) == 1:
                textInput = self.raw_dataset.labels_to_text[int(datum["label"])]
                label = 1
            else:
                randomID = random.choice(list(set(range(0, 364)) - set([int(datum["label"])])))
                textInput = self.raw_dataset.labels_to_text[randomID]
                label = 0
        else:
            textInput = self.raw_dataset.labels_to_text[int(datum["classindex"])]
            label = int(datum["label"])
         # Get image info
        pathname = self.raw_dataset.imgfeatpath + img_id + ".tsv"

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

                feats = data["features"].copy()
                obj_num = data["num_boxes"] 
                boxes = data["boxes"].copy()  
                objectsid = data["objects_id"]
                objectsconfid = data["objects_conf"]    # Read image features
                img_h, img_w = data['img_h'], data['img_w']

        #img_info = self.imgid2img[img_id]
        #obj_num = img_info['num_boxes']
        #boxes = img_info['boxes'].copy()
        #feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        

        return textInput, torch.tensor(feats), torch.tensor(boxes), torch.tensor(label), torch.tensor(objectsid), img_id


# class NLVR2Evaluator:
#     def __init__(self, dataset: NLVR2Dataset):
#         self.dataset = dataset

#     def evaluate(self, quesid2ans: dict):
#         score = 0.
#         for quesid, ans in quesid2ans.items():
#             datum = self.dataset.id2datum[quesid]
#             label = datum['label']
#             if ans == label:
#                 score += 1
#         return score / len(quesid2ans)

#     def dump_result(self, quesid2ans: dict, path):
#         """
#         Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
#         NLVR2 CSV file requirement:
#             Each line contains: identifier, answer

#         :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
#         :param path: The desired path of saved file.
#         :return:
#         """
#         with open(path, 'w') as f:
#             for uid, ans in quesid2ans.items():
#                 idt = self.dataset.id2datum[uid]["identifier"]
#                 ans = 'True' if ans == 1 else 'False'
#                 f.write("%s,%s\n" % (idt, ans))

