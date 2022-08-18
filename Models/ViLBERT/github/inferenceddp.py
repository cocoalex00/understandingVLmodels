# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ViLBERT model to image classification.

# imports 
from re import T
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import traceback

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
from vilbert.datasets.ImageClassificationDataset import  ImageClassificationDataset


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from distribuido import setup_for_distributed, save_on_master, is_main_process

from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)
from torch.nn import CrossEntropyLoss
try:
    import torch.cuda.amp as amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

gpu_list = "0,1,2,3"
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="gloo",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


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
    if not os.path.exists(folder_path + "/"):
        os.makedirs(folder_path+"/")
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

    n_gpu = torch.cuda.device_count() 
    # check for multiple gpus and spawn processes
    if n_gpu > 1:
        init_distributed()
        DISTRIBUTED = True
        print(f"spawned process {int(os.environ['RANK'])}/ {int(os.environ['WORLD_SIZE'])}")
    else:
        DISTRIBUTED = False


    # initialise the sigterm handler
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Pipeline everything by using an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
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
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/places365_val.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--tsv_test",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (train)"
    )
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
        "--batch_size",
        type=int,
        default=2,
        help="The number of samples in each batch.",
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
    if dist.is_available() and dist.is_initialized():
        device = "cuda:" + str(dist.get_rank())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if is_main_process() or not DISTRIBUTED:
        print("the device being used is: " + str(device))
        print("number of gpus available: " + str(n_gpu))
        print(f"Using mixed precision training: {APEX_AVAILABLE}")

    if is_main_process() or not DISTRIBUTED:
        print("## Loading the dataset ##")

    TestDataset = ImageClassificationDataset(args.annotTest,  args.tsv_test, 36,False)


    
    if DISTRIBUTED:
        testSampler = DistributedSampler(dataset=TestDataset, shuffle=True)   
       

        testDl = DataLoader(
            dataset= TestDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler=testSampler
        )
    else:
        testDl = DataLoader(
            dataset= TestDataset,
            batch_size= args.batch_size,
            shuffle= True,
            pin_memory=True,
        )




    if is_main_process()  or not DISTRIBUTED:
        print("## Dataset loaded succesfully ##")


    #################################################### MODEL STUFF ################################################################
    # Load the Model
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the Model ##")

    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = VILBertForImageClassification(args.config_file, args.num_labels, args.from_pretrained, device)
    # Only train the last classifier


     ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.pretrained,"VilbertFineTuned.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.pretrained,"VilbertFineTuned.pth"))

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if DISTRIBUTED:
            model.module.load_state_dict(checkpoint['model_checkpoint'], strict= False, map_location=device)
        else:
            model.load_state_dict(checkpoint['model_checkpoint'], strict= False)

        if is_main_process() or not DISTRIBUTED:
            print("Checkpoint loaded")
    else:
        if is_main_process() or not DISTRIBUTED:
            print("No checkpoint found")
           # sys.exit(1)

    

    if is_main_process() or not DISTRIBUTED:
        print("## Model Loaded ##")


    #################################################### Wrap up model with DDP and send to device #################################################
    if DISTRIBUTED:
        model =  nn.SyncBatchNorm.convert_sync_batchnorm(model)
        local_rank = int(os.environ["LOCAL_RANK"])
        model.to(local_rank)
        model = nn.parallel.DistributedDataParallel(model,device_ids = [local_rank])
        if is_main_process():
            print("Data parallelization activated")
    else:
        model.to(device)


    model.requires_grad_(False)

    ############################################### loss functions, optims, schedulers ##################################################
    criterion = CrossEntropyLoss()


    



    # Lists to keep track of nice stuff like loss and lr 
    testLoss = []
    accuracyTest = []

    # # Check for available checkpoints 
    # if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth")):
    #     print(f"Checkpoint found, loading")
    #     checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth"))
    #     start_epoch = checkpoint['current_epoch']
    #     learningRate = checkpoint['lr']
    #     trainingLoss = checkpoint['Tloss']
    #     valLoss = checkpoint['Vloss']

    #     # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
    #     if isinstance(model, nn.DataParallel):
    #         model.module.load_state_dict(checkpoint['model_checkpoint'])
    #     else:
    #         model.load_state_dict(checkpoint['model_checkpoint'])
    #     optimizer.load_state_dict(checkpoint['optimizer_checkpoint'])
    #     warmupScheduler.load_state_dict(checkpoint['warmup_checkpoint'])
    #     reducelrScheduler.load_state_dict(checkpoint['scheduler_checkpoint'])


    #     print("Checkpoint loaded")
    # else:
    #     print("No checkpoint found, starting fine tunning from base pre-trained model")


    # Data related stuff
                

    # Training loop ####################################################################################################################################
    
    print("***** Running inference *****")
    print("  Batch size: ", args.batch_size)

    # Wrap the training loop in a try, finally block to properly catch the sigterm signal
    try:

        ########################################### Validation  #####################################################

        model.eval() # Get model in eval mode

        running_loss_val = 0 # Keep track of avg loss for each epoch (val)
        correctOnes = 0 # Keep track of avg loss for each epoch (val)
        for i, batch in enumerate(testDl):

            batch = [t.cuda(device=device, non_blocking=True) for t in batch]
            textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, target = batch
            labels = torch.argmax(target,1)
            labels = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


            ### Forward pass ###
            with amp.autocast(): # Cast from f32 to f16 
                outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask)
            
                # Calculate batch loss
                loss = criterion(outputs,labels)

            # Add loss to list (val)
            lossitem = loss.detach()

            if DISTRIBUTED:
                dist.all_reduce(lossitem)
                lossitem = lossitem.item()/n_gpu
            running_loss_val += lossitem


            top1 = torch.topk(outputs,1)[1].squeeze(1)
            correctsval = (torch.eq(top1,labels).sum()).detach()
            correctOnes += correctsval

            if is_main_process() or not DISTRIBUTED:
                print(lossitem)
    
        if DISTRIBUTED: 
            dist.all_reduce(correctOnes)
            accuracy = correctOnes.item()/len(TestDataset)
        else:
            accuracy = correctOnes.item()/len(TestDataset)

        #accuracy_running_val += accuracyitem
        testLoss = running_loss_val / len(testDl)

        # Update the progress bar 
        if is_main_process() or not DISTRIBUTED:
            print(f"inference loss: {testLoss} / accuracy: {accuracy}")
    
    
    ############################################ Do when exception happens (eg sigterm) #############################################
    except :
        traceback.print_exc()
        # when sigterm caught, save checkpoint and exit
        if DISTRIBUTED:
                # wait for all gpus to get to here before saving
            dist.barrier()
        sys.exit(0)

    
    ###################################################### Create plots and save them ########################################################
    if is_main_process() or not DISTRIBUTED:

        # Also save the lists and stats in a csv to create other plots if needed
        with open(f"{args.output_dir}/ALBEFInference.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["inference loss:"])
            write.writerow(testLoss)
            write.writerow(["inference Accuracy"])
            write.writerow(accuracy)

if __name__=="__main__":
    main()