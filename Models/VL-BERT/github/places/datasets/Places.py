from cProfile import label
import os
import time
import jsonlines
import json
import _pickle as cPickle
from PIL import Image
from copy import deepcopy
import csv
import base64
import numpy as np

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer, BasicTokenizer


from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.mask import generate_instance_mask
from common.nlp.misc import get_align_matrix
from common.utils.misc import block_digonal_matrix
from common.nlp.misc import random_word_with_token_ids
from common.nlp.roberta import RobertaTokenizer
import sys
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


class Places(Dataset):
    def __init__(self, ann_file, tsv_path, root_path,
                 basic_tokenizer=None, tokenizer=None, mask_size=(14, 14),
                 aspect_grouping=False, basic_align=False, seq_len=2,
                 **kwargs):
        """
        Places365 Dataset
        
        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param tsv_path: path to the tsv file with the pre-computed image features
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param only_use_relevant_dets: filter out detections not used in query and response
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param basic_align: align to tokens retokenized by basic_tokenizer
        :param kwargs:
        """
        super(Places, self).__init__()

        self.seq_len = seq_len
        self.tsv_path = tsv_path
        self.ann_file = ann_file
        self.root_path = root_path
        self.aspect_grouping = aspect_grouping
        self.basic_align = basic_align
        print('Dataset Basic Align: {}'.format(self.basic_align))
        self.mask_size = mask_size


        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained('bert-base-uncased')

        self.database = self.load_annotations(self.ann_file)
        #self.img_data = self.load_obj_tsv(self.tsv_path)

        # transforming from list to dictionary for better accessing
        # self.imgid2img = {}
        # for img_datum in self.img_data:
        #     self.imgid2img[img_datum['img_id']] = img_datum


        self.id2datum = {
            datum['img_name']: datum
            for datum in self.database
        }

        if self.aspect_grouping:
            assert False, "Not support aspect grouping now!"
            self.group_ids = self.group_aspect(self.database)

    def load_annotations(self, ann_file):
        tic = time.time()
        database = []

        # ignore or not find cached database, reload it from annotation file
        print('loading database from {}...'.format(ann_file))
        tic = time.time()

        # with jsonlines.open(ann_file) as reader:
        #     for ann in reader:
        #         db_i = {
        #             'id': ann['id'],
        #             'img_name': ann['img_name'],
        #             'img_path': ann['img_path'],
        #             'label': ann['label']
        #         }
        #         database.append(db_i)
        database.extend(json.load(open(self.ann_file)))

        print('Done (t={:.2f}s)'.format(time.time() - tic))

        return database

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

 

    def __getitem__(self, index):
        # self.person_name_id = 0
        idb = deepcopy(self.database[index])
        img_name = idb["img_path"].replace("-","/") 
        image_path = os.path.join(self.root_path, img_name)
        label1 = idb["label"]

        pathname = self.tsv_path + idb["img_name"] + ".tsv"
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

                # features = data["features"]
                # num_boxes = data["num_boxes"] 
                # boxes = data["boxes"]      # Read image features


        #visual_features = deepcopy(self.imgid2img[idb["img_name"]])
                w0,h0 = data["img_w"], data["img_h"]

                boxes_features = torch.as_tensor(data['features'], dtype=torch.float32).reshape((data['num_boxes'], -1))

                boxes = torch.as_tensor(data['boxes'], dtype=torch.float32).reshape((data['num_boxes'], -1))

        # Add full image as box
        image_box = torch.as_tensor([[0.0, 0.0, w0 - 1, h0 - 1]])
        boxes = torch.cat((image_box, boxes), dim=0)
        image_box_feature = boxes_features.mean(0, keepdim=True)
        boxes_features = torch.cat((image_box_feature, boxes_features), dim=0)

        im_info = torch.tensor([w0, h0, 1.0, 1.0])


        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # Pre-computed features mean no image
        # We have to do some weird checking in order for the paths to work 
        list_root = self.root_path.split("/")
        imgfile = image_path.split("/")
        otherstuff = ["user", "HS400", "ah02299", "lxmert", "totest"]

        # The directory structure in the train folder is:
        #   - train/
        #       - a/
        #           - airport/
        #               - 000001.jpg

        imgfile = [directory for directory in imgfile if directory not in list_root and directory not in otherstuff] # get rid of all the parts of the img_name which are already in the root path
        imgfile = "/".join(imgfile) # join the rest of the name with slashes to make the rest of the path 
        image_path = os.path.join(self.root_path,imgfile)
        #print(img)
        #image = np.array(self._load_image(image_path))

        # Text input 
        text = "[CLS] [SEP]"
        q_retokens = self.tokenizer.tokenize(text)
        q_ids = self.tokenizer.convert_tokens_to_ids(q_retokens)
        print(q_ids)





        # concat box feature to box
        boxes = torch.cat((boxes, boxes_features), dim=-1)



        return boxes,im_info, q_ids, label1 #  image, 

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path)
        else:
            return Image.open(path)

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)

    # Added by Alejandro Hernandez Diaz
    def load_precomputed_boxes(self, box_file):
        in_data = {}
        with open(box_file, "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                item['image_id'] = item['image_id']
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in (['boxes', 'features'] if self.with_precomputed_visual_feat else ['boxes']):
                    item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                                dtype=np.float32).reshape((item['num_boxes'], -1))
                in_data[item['image_id']] = item
        return in_data

    def load_obj_tsv(self,fname, topk=None):
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


    @property
    def data_names(self):
        return ['image','boxes','textual input','label']

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())
