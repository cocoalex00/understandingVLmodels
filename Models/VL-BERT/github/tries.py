import torch.nn as nn 
from torch.optim import Adam

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm
from places.datasets.Places import Places
from torch.utils.data import DataLoader
from transformers.optimization import AdamW

from places.modules.resnet_vlbert_for_image_classification import ResNetVLBERT
import torch
import numpy as np

#om transformers import default_data_collator

MAX_VQA_LENGTH = 2
NUM_CLASSES = 365
LR= 0.000002
NUM_EPOCHS= 300
BATCH_SIZE = 2
WEIGHT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/snap/pretrained/model"
ANNOT = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/VL-BERT/github/data/Places/train.json"
TSV = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/VL-BERT/github/data/Places/valid_obj36.tsv"
CONFIG= "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/VL-BERT/github/cfgs/Places/places_base.yaml"
ROOT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/images/val_256/"
from pretrain.function.config import config, update_config
if __name__ == "__main__":
    # Load dataset and DataLoader
    dataset = Places(ANNOT,TSV,ROOT_PATH)
    data_loader = DataLoader(
        dataset, batch_size=2,
        shuffle=False, num_workers=12,
         pin_memory=True
    )

    boxes, im_info, text_ids, label = next(iter(data_loader))
    text_ids = torch.stack(text_ids)

    # Update the config file of the model according to the yaml file of the task
    update_config(CONFIG)

    # Initialise image classification model and its weights
    model = ResNetVLBERT(config)
    model.init_weight()  

    #Freeze the weights up to the final mlp for fine tuning
    model.image_feature_extractor.requires_grad_(False)
    #model.vlbert.requires_grad_(False)

    # for param in model.parameters():
    #     if param.requires_grad == True: print(param) 
    # model in train mode
    model.train()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=LR)
    #print(model)
     
    # Run training
    print("***** Running training *****")
    print("  Num epochs: ", NUM_EPOCHS)
    print("  Batch size: ", BATCH_SIZE)

    for i in tqdm(range(NUM_EPOCHS)):
        output = model.train_forward(None,boxes,im_info,text_ids)
        loss = criterion(output[1],label)
        loss.backward()
        optim.step()
        print(loss)