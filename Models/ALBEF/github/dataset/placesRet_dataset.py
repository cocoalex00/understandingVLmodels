import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
from torchvision import transforms
from transformers import BertTokenizer
import pandas as pd
import random


class places365(Dataset):
    def __init__(self, ann_file,labels_path,image_root, transform = None,val = False):        
      
        self.ann = json.load(open(ann_file,'r'))
        
        # # Transforms 
        # train_transform = transforms.Compose([                        
        #     transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
        #                                       'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        #     transforms.ToTensor(),
        #     normalize,
        # ])  
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transform = transforms.Compose([
            transforms.Resize((256,256),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])   
        self.image_root = image_root
        self.max_words = 7
        self.val = val

        self.labels_to_text = pd.read_csv(labels_path, header=None, delimiter = "/")[0].values.tolist()

        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]

        # We have to do some weird checking in order for the paths to work 
        list_root = self.image_root.split("/")
        imgfile = ann['img_path'].split("-")
        otherstuff = ["user", "HS400", "ah02299", "lxmert", "totest"]

        # The directory structure in the train folder is:
        #   - train/
        #       - a/
        #           - airport/
        #               - 000001.jpg

        imgfile = [directory for directory in imgfile if directory not in list_root and directory not in otherstuff] # get rid of all the parts of the img_name which are already in the root path
        imgfile = "/".join(imgfile) # join the rest of the name with slashes to make the rest of the path 

        image_path = self.image_root + imgfile      
        image0 = Image.open(image_path).convert('RGB')   
        image0 = self.transform(image0)   
                


        if self.val:
            sentence = self.labels_to_text[int(ann["classindex"])]
            label = ann["label"]
        else:
            if random.choice([0,1]) == 1:
                sentence = self.labels_to_text[int(ann["label"])]
                label = 1
            else:
                randomID = random.choice(list(set(range(0, 364)) - set([int(ann["label"])])))
                sentence = self.labels_to_text[randomID]
                label = 0


        return image0, sentence, label
