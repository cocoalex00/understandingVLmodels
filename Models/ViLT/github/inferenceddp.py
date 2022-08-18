# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted LXMERT model to image classification.

# imports 
import enum
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from vilt.modules import ViLTransformerSS
from vilt.datasets.Places_dataset import Places365
import functools

import traceback


from vilt.config import ex

import csv
import signal
import sys 
import os
import torch.nn as nn
import numpy as np
import random 
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import  get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, BertTokenizer
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

from distribuido import setup_for_distributed, save_on_master, is_main_process


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
            backend="nccl",
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
        torch.save(obj=checkpoint, f=f"{folder_path}/ViLTFineTuned.pth")
    else:
        torch.save(obj=checkpoint, f=f"{folder_path}/checkpointViLT.pth")
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

@ex.automain
def main(_config):
    n_gpu = torch.cuda.device_count() 
    # check for multiple gpus and spawn processes
    if n_gpu > 1:
        init_distributed()
        DISTRIBUTED = True
        print(f"spawned process {int(os.environ['RANK'])}/ {int(os.environ['WORLD_SIZE'])}")
    else:
        DISTRIBUTED = False

    # Pipeline everything by using an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        default="/vol/teaching/HernandezDiazProject/understandingVLmodels/Models/ViLT/github/pretrained/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="Path to the checkpoint file containing the model's weights and stuff",
    )
    parser.add_argument(
        "--data_path",
        default="/vol/teaching/HernandezDiazProject/Data/arrowfiles",
        type=str,
        help="Path to the folder where the dataset file (.arrow) lives",
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
        default="/vol/teaching/HernandezDiazProject/understandingVLmodels/Experiments/ViLT/imgClf/outfull",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=33,
        help="The number of samples in each batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for initialisation in multiple GPUs"
    )
    parser.add_argument

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

    # if n_gpu > 1: 
    #     args.batch_size = args.batch_size * n_gpu   # Augment batch size if more than one gpu is available

    
    #################################################### Dataset ##################################################################
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the dataset ##")

    # Load the custom data collator as well as the tokenizer 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = DataCollatorForLanguageModeling(tokenizer,False)

    TestDataset = Places365(args.data_path, split="test")   ####### Change this to train for the HTCONDOR!!!!!!!

    if DISTRIBUTED:
        testSampler = DistributedSampler(dataset=TestDataset, shuffle=True)   
        # create the two custom collator functions
        collateTest = functools.partial(
                TestDataset.collate, mlm_collator=collator,
            )
        testSampler = DistributedSampler(dataset=TestDataset, shuffle=True)   
       

        testDl = DataLoader(
            dataset= TestDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler=testSampler,
            collate_fn=collateTest,
        )
    else:
        collateTest = functools.partial(
                TestDataset.collate, mlm_collator=collator,
            )
        testDl = DataLoader(
            dataset= TestDataset,
            batch_size= args.batch_size,
            shuffle= True,
            pin_memory=True,
            collate_fn=collateTest,
        )

    if is_main_process()  or not DISTRIBUTED:
        print("## Dataset loaded succesfully ##")


    #################################################### MODEL STUFF ################################################################
    # Load the Model
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the Model ##")
    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = ViLTransformerSS(_config,"places")

    ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.pretrained,"ViLTFineTuned.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.pretrained,"ViLTFineTuned.pth"))

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if DISTRIBUTED:
            model.module.load_state_dict(checkpoint['model_checkpoint'], strict= False)
        else:
            model.load_state_dict(checkpoint['model_checkpoint'], strict= False)

        if is_main_process() or not DISTRIBUTED:
            print("Checkpoint loaded")
    else:
        if is_main_process() or not DISTRIBUTED:
            print("No checkpoint found, starting fine tunning from base pre-trained model")


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



    

    ############################################### loss functions, optims, schedulers ##################################################
    criterion = CrossEntropyLoss()

    # Lists to keep track of nice stuff like loss and lr 
    testLoss = []
    accuracyTest = []


    # # Check for available checkpoints 
    # if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointViLT.pth")):
    #     print(f"Checkpoint found, loading")
    #     checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointViLT.pth"))
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
 



    print("***** Running inference *****")
    print("  Batch size: ", args.batch_size)

    # Wrap the training loop in a try, finally block to properly catch the sigterm signal
    try:

        ########################################### Validation  #####################################################

        model.eval() # Get model in eval mode

        running_loss_val = 0 # Keep track of avg loss for each epoch (val)
        correctOnes = 0 # Keep track of avg loss for each epoch (val)
        for i, batch in enumerate(testDl):

            label = torch.tensor(batch["label"]).to(device).squeeze()
            for i in batch.keys():
                if type(batch[i]) != list:
                    batch[i] = batch[i].to(device)

            batch["image"] = [batch["image"][0].to(device)]
        

            ### Forward pass ###
            with amp.autocast(): # Cast from f32 to f16 
                output = model(batch)

                # Calculate batch loss
                loss = criterion(output["imgcls_logits"],label)
    

            # Add loss to list (val)
            lossitem = loss.detach()

            if DISTRIBUTED:
                dist.all_reduce(lossitem)
                lossitem = lossitem.item()/n_gpu
            running_loss_val += lossitem


            top1 = torch.topk(output["imgcls_logits"],1)[1].squeeze(1)
            correctsval = (torch.eq(top1,label).sum()).detach()
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
        

