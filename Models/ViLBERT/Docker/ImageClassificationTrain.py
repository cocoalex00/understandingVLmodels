# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ViLBERT model to image classification.

from email.policy import default
import torch
import torch.distributed as dist
import torch.nn as nn


import numpy as np
import random 
from tqdm import tqdm


import argparse

from vilbert.vilbert import BertConfig
from vilbert.vilbertImageClassification import VILBertForImageClassification
from vilbert.datasets.ImageClassificationDataset import get_data_loader 

from pytorch_transformers.optimization import (
    AdamW
)

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
import json

import matplotlib.pyplot as plt


def main():

    # Pipeline everything by using an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(    
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--data_annotations",
        type=str,
        default="data/val.jsonline",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--lmdb",
        type=str,
        default="data/prueba.lmdb/",
        help="Path to the lmdb file containing the features of the dataset"
    )
    parser.add_argument(
        "--local_rank",
        type= int,
        default=-1,
        help="local_rank for distributed training",
    )
    parser.add_argument(
        "--num_labels",
        default=365,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=0.00002,
        help="The base learning rate to be used with the optimizer (default =0.00002)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for initialisation in multiple GPUs"
    )
    # Get all arguments 
    args = parser.parse_args()

    # Need to set up a seed so that all the models initialised in the different GPUs are the same
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          # This may allow training on the newer gpus idk



    # CUDA check 
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu") 
    #device = torch.device("cpu")  # Assign device to run the code
    n_gpu = torch.cuda.device_count()                                       # Check the number of GPUs available
    
    print("the device being used is: " + str(device))
    print("number of gpus available: " + str(n_gpu))

    # Check to only run certain parts of the code in the primary gpu
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    
    # Load the dataset
    if default_gpu:
        print("## Loading the dataset ##")
    dataloader = get_data_loader(args.data_annotations,args.lmdb)
    if default_gpu:
        print("## Dataset loaded succesfully ##")

    # Load the Bert weights
    if default_gpu:
        print("## Loading the Model ##")
    #bert_weight_name = json.load(
    #    open("config/bert-base-uncased_weight_name.json", "r")
    #)

    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = VILBertForImageClassification(args.config_file, args.num_labels, args.from_pretrained)
    model.vilbertBase.requires_grad_(False)
    if default_gpu:
        print("## Model Loaded ##")
    # Paralellization in multiple gpus
    #model = nn.DataParallel(model)
    torch.cuda.set_device(device)
    model.cuda(device)

    # Hyperparameters
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), args.lr)

    lr_scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )
    

    # Training loop
    if default_gpu:
        print("***** Running training *****")
        print("  Num epochs: ", args.num_epochs)
        print("  Batch size: ", args.batch_size)
        
    #Model.train() change to inside the loop when using validation
    model.train()
    #  for i, batch in enumerate(dataloader):
    #     print(np.array(batch).shape)
    #batch = next(iter(dataloader))
    
    
   
    losses = []
    for epoch in tqdm(range(args.num_epochs),desc="Epoch"):
        for i, batch in enumerate(dataloader):
            batch = [t.cuda(device=device, non_blocking=True) for t in batch]

            textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, target = batch
            labels = torch.argmax(target,1)
            labels = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)

            outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
            


            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)


            #losses.append(loss)
            #if default_gpu:
            #plt.plot(loss)

if __name__=="__main__":
    main()
