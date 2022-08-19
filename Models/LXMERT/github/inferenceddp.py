# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted LXMERT model to image classification.

# imports 
import enum
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import torch

from src.tasks.places_data import PlacesDataset, PlacesTorchDataset
from torch.utils.data import DataLoader
from src.tasks.imgclassif_model import imgClassifModel

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
    # Save to checkpoint file, use different name for final checkpoint #
    if final:
        torch.save(obj=checkpoint, f=f"{folder_path}/LXMERTFineTuned.pth")
    else:
        torch.save(obj=checkpoint, f=f"{folder_path}/checkpointLXMERT.pth")
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

    # Pipeline everything by using an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/LXMERT/imgClf/out50",
        type=str,
        help="PATH to the .pth file contatining the pre-trained weights. Ojo, the function loads it like 'Path + _LXRT.pth' so omit that part"
    )
    #### Annotations and TSV files 
    parser.add_argument(
        "--annotTest",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/Utilities/jsonfiles/places365_test_alexsplit.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--local_rank",
        type=int
    )

    parser.add_argument(
        "--tsv_test",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/train/",
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
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/LXMERT/imgClf/out50",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
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

    TestDataset = PlacesTorchDataset(PlacesDataset(args.annotTest, args.tsv_test))
    
    if DISTRIBUTED:
        testsampler = DistributedSampler(dataset=TestDataset, shuffle=True)   

        testDl = DataLoader(
            dataset= TestDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler=testsampler
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
    model = imgClassifModel(args.num_labels)


    if os.path.exists(os.path.join(args.checkpoint_dir,"LXMERTFineTuned.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.checkpoint_dir,"LXMERTFineTuned.pth"), map_location=device)

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if DISTRIBUTED:
            model.load_state_dict(checkpoint['model_checkpoint'], strict= False)
        else:
            model.load_state_dict(checkpoint['model_checkpoint'], strict= False)
        if is_main_process() or not DISTRIBUTED:
            print("Checkpoint loaded")
    else:
        if is_main_process() or not DISTRIBUTED:
            print("No checkpoint found")

    

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



        ##################################################### Checkpoint or pre-trained ###################################################


    
    print("***** Running inference *****")
    print("  Batch size: ", args.batch_size)

    # Wrap the training loop in a try, finally block to properly catch the sigterm signal
    try:
        
        ########################################### Validation  #####################################################

        model.eval() # Get model in eval mode

        running_loss_val = 0 # Keep track of avg loss for each epoch (val)
        correctOnes = 0 # Keep track of avg loss for each epoch (val)
        for i, batch in enumerate(testDl):

                
            txt, imgfeats, boxes, labels, objectsid, img_id = batch
            imgfeats, boxes=  imgfeats.cuda(), boxes.cuda()

            labelsTensor = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


            ### Forward pass ###
            with amp.autocast(): # Cast from f32 to f16 
                outputs = model(imgfeats,boxes,txt)

                # Calculate batch loss
                loss = criterion(outputs,labelsTensor)

            # Add loss to list (val)
            lossitem = loss.detach()

            if DISTRIBUTED:
                dist.all_reduce(lossitem)
                lossitem = lossitem.item()/n_gpu
            running_loss_val += lossitem


            top1 = torch.topk(outputs,1)[1].squeeze(1)
            correctsval = (torch.eq(top1,labelsTensor).sum()).detach()
            correctOnes += correctsval

            if is_main_process() or not DISTRIBUTED:
                print(f"batch {i}: loss {lossitem}")
    
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
    
    
#
    except Exception as e:
        # when sigterm caught, save checkpoint and exit
        print(e)
        # when sigterm caught, save checkpoint and exit
        if DISTRIBUTED:
                # wait for all gpus to get to here before saving
            dist.barrier()
        sys.exit(0)



    

    ###################################################### Create plots and save them ########################################################
    if is_main_process() or not DISTRIBUTED:
        # Also save the lists and stats in a csv to create other plots if needed
        with open(f"{args.output_dir}/LXMERTInference.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["inference loss:"])
            write.writerow(testLoss)
            write.writerow(["inference Accuracy"])
            write.writerow(accuracy)
if __name__=="__main__":
    main()
