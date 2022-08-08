from src.utils import load_obj_tsv
from src.tasks.places_data import PlacesDataset, PlacesTorchDataset
from src.param import args
from src.lxrt.entry import LXRTEncoder
from torch.utils.data import DataLoader
from src.tasks.imgclassif_model import imgClassifModel
import torch.nn as nn 
from torch.optim import Adam

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm

import torch

MAX_VQA_LENGTH = 2
NUM_CLASSES = 365
LR= 0.000002
NUM_EPOCHS= 300
BATCH_SIZE = 5
WEIGHT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/snap/pretrained/model"

if __name__ == "__main__":
    model = imgClassifModel(
        num_classes=365
    ).cuda()
    print(model)
    # Load pre-trained weights
    model.lxrt_encoder.load(WEIGHT_PATH)

    dataset = PlacesDataset("/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/data/Places/train.json", "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/data/Places/valid_obj36.tsv")
    datasetU =PlacesTorchDataset(dataset)
    data_loader = DataLoader(
        datasetU, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=12,
         pin_memory=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optim = Adam(model.parameters(), lr=LR)

    batch = next(iter(data_loader))
    
 
    txt, imgfeats, boxes, label, objectsid, img_id, objectsconf = batch
    print(imgfeats[1].shape)
    label = torch.from_numpy(to_categorical(label, num_classes=365))
    imgfeats, boxes, label =  imgfeats.cuda(), boxes.cuda(), label.cuda()

    for rawids,confi in zip(objectsid,objectsconf):

        # indexeskept =[]
        # index = 0
        # for confidence in confi:
        #     if confidence > 0.5:
        #         indexeskept.append(index)
        #     index += 1
        

        # print("object id--------------------")
        # print(rawids[indexeskept])
        # print("wwith confidence-------------")
        # print(confi[indexeskept])
    
        print("object id--------------------")
        print(rawids)
        print("wwith confidence-------------")
        print(confi)
    print("image id--------------------")
    print(img_id)
 
    # print("***** Running training *****")
    # print("  Num epochs: ", NUM_EPOCHS)
    # print("  Batch size: ", BATCH_SIZE)

    # for i in tqdm(range(NUM_EPOCHS), desc="epochs"):

    #     output = model(imgfeats,boxes,txt)
    #     loss = criterion(output,label)
    #     loss.backward()
    #     optim.step()
    #     print(loss)