# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ViLBERT model to image classification.

# imports 
import matplotlib.pyplot as plt
from regex import P
import torch

import csv
import signal
import sys 
import os
import torch.nn as nn
import numpy as np
import random 
from tqdm import tqdm
import argparse
from vilbert.vilbertImageClassification import VILBertForImageClassification
from vilbert.datasets.ImageClassificationDataset import get_data_loader 

import torch.nn.functional as F

try:
    import torch.cuda.amp as amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False



def save_checkpoint(folder_path,model_checkpoint,optimizer_checkpoint,warmup_checkpoint,scheduler_checkpoint,Tloss_checkpoint,Vloss_checkpoint,lr_checkpoint, current_epoch, final: bool= False):
    ''' This function saves a checkpoint of the training process to be loaded later
        - model_checkpoint: model's state dictionary
        - optimizer_checkpoint: optimizers's state dictionary
        - warmup_checkpoint: warmup scheduler's state dictionary
        - scheduler_checkpoint: scheduler's state dictionary
        - Tloss_checkpoint: list of training losses up to that point
        - Vloss_checkpoint: list of validation losses up to that point
        - lr_checkpoint: list of past learning rates
        - current_epoch: current training epoch
    '''
    print("--Saving checkpoint--")
    # Create the checkpoint dictionary
    checkpoint = {
        'model_checkpoint': model_checkpoint,
        'optimizer_checkpoint': optimizer_checkpoint,
        'warmup_checkpoint': warmup_checkpoint,
        'scheduler_checkpoint': scheduler_checkpoint,
        'Tloss': Tloss_checkpoint,
        'Vloss': Vloss_checkpoint,
        'lr': lr_checkpoint,
        'current_epoch': current_epoch
    }
    # Save to checkpoint file, use different name for final checkpoint #
    if final:
        torch.save(obj=checkpoint, f=f"{folder_path}/VilbertFineTuned.pth")
    else:
        torch.save(obj=checkpoint, f=f"{folder_path}/checkpointVilbert.pth")
    print("--Checkpoint saved--")



# For HTCondor
def sigterm_handler(signal,frame):
    ''' This funciton detects the sigterm signal sent to the script when the job is about to be
        evicted (HTCONDOR)
    ''' 
    # catch sigterm and raise system exit 
    print("Sigterm caught!")
    sys.exit(0)



#### Main Script ###
def main():
    # initialise the sigterm handler
    signal.signal(signal.SIGTERM, sigterm_handler)

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
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ViLBERT/github/config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    #### Annotations and TSV files 
    parser.add_argument(
        "--annotTest",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/images/totest/train.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )

    parser.add_argument(
        "--tsv_test",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/Dataset_Utilities/valid_obj36.tsv",
        help="Path to the tsv file containing the features of the dataset (train)"
    )
    ####
    parser.add_argument(
        "--num_labels",
        default=365,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/ViLBERT/imgClf/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/ViLBERT/imgClf/checkpoints",
        type=str,
        help="The output directory where the training checkpoints will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The number of samples in each batch.",
    )


    # Get all arguments 
    args = parser.parse_args()



    # CUDA check 
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu") 
    n_gpu = torch.cuda.device_count()                                       # Check the number of GPUs available


    print("the device being used is: " + str(device))
    print("number of gpus available: " + str(n_gpu))

    if n_gpu > 1: 
        args.batch_size = args.batch_size * n_gpu   # Augment batch size if more than one gpu is available

    
    # Load the dataset
    print("## Loading the dataset ##")

    testDL = get_data_loader(args.annotTest, args.tsv_test, args.batch_size,test=True)


    print("## Dataset loaded succesfully ##")


    ##### MODEL STUFF #####
    # Load the Model
    print("## Loading the Model ##")
    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = VILBertForImageClassification(args.config_file, args.num_labels, args.from_pretrained)
    # Only train the last classifier
    model.requires_grad_(False)


    

    print("## Model Loaded ##")

    # Data paralellization in multiple gpus if more than one is available (only in one node, multiple is too hard f***)
    if n_gpu > 1:
        device = torch.device("cuda")
        model = nn.DataParallel(model)
        model.to(device)
        print("Data parallelization activated")
    else:
        model.to(device)


    # Lists to keep store the model's predicitions
    Predictions = []

    # Check for available checkpoints 
    if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth")):
        print(f"Checkpoint found, loading")
        checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth"))

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_checkpoint'])
        else:
            model.load_state_dict(checkpoint['model_checkpoint'])


        print("Checkpoint loaded")
    else:
        print("No checkpoint found, starting fine tunning from base pre-trained model")






    # Training loop ####################################################################################################################################

    print("***** Running Inference *****")
    print(len(testDL))

    allPreds = [] # Keep track of avg loss for each epoch (train)
    with open(f"{args.output_dir}/predictions.txt", "w") as f:
        write = csv.writer(f)
        for i, batch in tqdm(enumerate(testDL)):

            # Data related stuff
            batch = [t.cuda(device=device, non_blocking=True) for t in batch]
            textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, target = batch
            


            ### Forward pass ###
            with amp.autocast(): # Cast from f32 to f16 
                outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)
                # run the logits through a softmax
                probabilities = F.softmax(outputs,1).cpu().detach().numpy()

            # Get top 5 predictions 
            Top5 = np.flip(np.argsort(probabilities),1)[:,0:5]

            
            
            write.writerows(Top5)

            ########################################### Validation  #####################################################


    # Also save the lists and stats in a csv to create other plots if needed 

        
if __name__=="__main__":
    main()
