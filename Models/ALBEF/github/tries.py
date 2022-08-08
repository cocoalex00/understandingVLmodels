import torch.nn as nn 
from torch.optim import Adam
import yaml

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm
from transformers import AdamW, BertTokenizer
from dataset.places_dataset import places365
from models.model_imgclf import ALBEF
from torch.utils.data import DataLoader

from torch.nn import CrossEntropyLoss

import torch
import numpy as np

from transformers import AdamW

#om transformers import default_data_collator

MAX_VQA_LENGTH = 2
NUM_CLASSES = 365
LR= 0.000002
NUM_EPOCHS= 300
BATCH_SIZE = 2
ANNOT = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/Dataset_Utilities/train.json"
CONFIG= "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/VL-BERT/github/cfgs/Places/places_base.yaml"
ROOT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/images/totest/"
CONFIG_FILE = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ALBEF/github/configs/Imgclf_places.yaml"
if __name__ == "__main__":

    config = yaml.load(open(CONFIG_FILE, 'r'), Loader=yaml.Loader)
    device = "cuda:0" if torch.cuda.is_available else "cpu"
    print(f"Running on device: {device}")

    dataset = places365(ANNOT, ROOT_PATH)
    dataloader = DataLoader(dataset= dataset,
                            batch_size=2,
                            pin_memory = True 
                        ) 
    image, text, label = next(iter(dataloader))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
    print(text_inputs)
    model = ALBEF(config=config, text_encoder="bert-base-uncased", tokenizer= tokenizer, num_labels = NUM_CLASSES).to(device)


    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    # Run training
    print("***** Running training *****")
    print("  Num epochs: ", NUM_EPOCHS)
    print("  Batch size: ", BATCH_SIZE)

    for i in tqdm(range(NUM_EPOCHS)):

        optimizer.zero_grad()

        output = model(image.to(device),text_inputs)
        loss = criterion(output,label.to(device))
        loss.backward()
        optimizer.step()
        print(loss)