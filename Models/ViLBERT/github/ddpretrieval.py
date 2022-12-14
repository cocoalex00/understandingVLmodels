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
import torch.nn.functional as F

import csv
import signal
import sys 
import os
import torch.nn as nn
import numpy as np
import random 
from tqdm import tqdm
import argparse
from vilbert.vilbertRetrievalPlaces import VILBertForRetrieval
from vilbert.datasets.ITretrievalDataset import ITretrievalPlaces


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
        "--bert_model",
        default="bert-base-uncased",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
    )
    parser.add_argument(
        "--from_pretrained",
        #default= "bert-base-uncased",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Models/ViLBERT/github/save/pretrained_model.bin",
        type=str,
    )
    parser.add_argument(    
        "--config_file",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Models/ViLBERT/github/config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    #### Annotations and TSV files 
    parser.add_argument(
        "--annotTrain",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/Utilities/jsonfiles/places365_train_alexsplit.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--annotVal",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/Utilities/jsonfiles/places365_retrieVal.json",
        help="Path to the json file containing the annotations of the dataset (validation)"
    )
    parser.add_argument(
        "--tsv_train",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/train/",
       # default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (train)"
    )
    parser.add_argument(
        "--tsv_val",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/val/",
        help="Path to the tsv file containing the features of the dataset (validation)",
    ####
    )
    parser.add_argument(
        "--num_labels",
        default=1,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/ViLBERT/ret/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/ViLBERT/ret/checkpoints",
        type=str,
        help="The output directory where the training checkpoints will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The number of samples in each batch.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="The number of epochs to train the model for.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=3e-4,
        help="The base learning rate to be used with the optimizer (default =0.00002)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for initialisation in multiple GPUs"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/Utilities/retrieval_labels.txt",
        help="random seed for initialisatio"
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

    trainDataset = ITretrievalPlaces(args.annotTrain,args.labels_path,  args.tsv_train, 36,False)
    valDataset = ITretrievalPlaces(args.annotVal,args.labels_path,args.tsv_val, 36,True)

    if DISTRIBUTED:
        trainsampler = DistributedSampler(dataset=trainDataset, shuffle=True)   
        valsampler = DistributedSampler(dataset=valDataset, shuffle=True) 

        trainDL = DataLoader(
            dataset= trainDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler=trainsampler
        )
        valDL = DataLoader(
            dataset= valDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler= valsampler
        )
    else:
        trainDL = DataLoader(
            dataset= trainDataset,
            batch_size= args.batch_size,
            shuffle= True,
            pin_memory=True
        )
        valDL = DataLoader(
            dataset= valDataset,
            batch_size= args.batch_size,
            shuffle= True,
            pin_memory=True
        )


    if is_main_process()  or not DISTRIBUTED:
        print("## Dataset loaded succesfully ##")


    #################################################### MODEL STUFF ################################################################
    # Load the Model
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the Model ##")

    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = VILBertForRetrieval(args.config_file,  args.from_pretrained, device)
    # Only train the last classifier
    model.vilbertBase.requires_grad_(False)

    

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
    criterion = nn.BCEWithLogitsLoss()
    learningrate = np.sqrt(n_gpu) * args.lr
    optimizer = AdamW(model.parameters(), learningrate)
    # Create gradient scaler for f16 precision
    scaler = amp.GradScaler(enabled=True)

    # learning rate schedulers 
    reducelrScheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )
    totalSteps = len(trainDL) * args.num_epochs
    warmupSteps = int(totalSteps * args.warmup_proportion)                            # Calculate the number of epochs to warm up for
    warmupScheduler = get_linear_schedule_with_warmup(optimizer= optimizer,
                                                    num_warmup_steps= warmupSteps,
                                                    num_training_steps = totalSteps
                                                    )
    



    # Lists to keep track of nice stuff like loss and lr 
    trainingLoss = []
    valLoss = []
    learningRate = []
    start_epoch = 0
    accuracyTrain = []
    valAccuracy = []

        ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth"), map_location=device)
        start_epoch = checkpoint['current_epoch']
        learningRate = checkpoint['lr']
        trainingLoss = checkpoint['Tloss']
        valLoss = checkpoint['Vloss']

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if DISTRIBUTED:
            model.module.load_state_dict(checkpoint['model_checkpoint'], strict= False)
        else:
            model.load_state_dict(checkpoint['model_checkpoint'], strict= False)
        optimizer.load_state_dict(checkpoint['optimizer_checkpoint'])
        warmupScheduler.load_state_dict(checkpoint['warmup_checkpoint'])
        reducelrScheduler.load_state_dict(checkpoint['scheduler_checkpoint'])

        if is_main_process() or not DISTRIBUTED:
            print("Checkpoint loaded")
    else:
        if is_main_process() or not DISTRIBUTED:
            print("No checkpoint found, starting fine tunning from base pre-trained model")

  

   

    # Progress bars
    pbarTrain = tqdm(range(start_epoch, args.num_epochs))

    # Training loop ####################################################################################################################################

    if is_main_process() or not DISTRIBUTED:
        print("***** Running training *****")
        print(f"  Num epochs left: {args.num_epochs - start_epoch}/{ args.num_epochs}")
        print("  Batch size: ", args.batch_size)

    # Wrap the training loop in a try, finally block to properly catch the sigterm signal
    try:
        for epoch in pbarTrain:
            ########################################### Training  #####################################################

            model.train() # Get model in training mode

            accuracy_running = 0 
            running_loss_train = 0 # Keep track of avg loss for each epoch (train)
            for i, batch in enumerate(trainDL):
                optimizer.zero_grad()               # Clear gradients of the optimizer
            

                 # Data related stuff
                batch = [t.cuda(device=device, non_blocking=True) for t in batch]
                textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, labels = batch
                #labels = torch.argmax(target,1)
                labels = torch.tensor([t.type(torch.float32)for t in labels], device=device)


                ### Forward pass ###
                with amp.autocast(): # Cast from f32 to f16 
                    outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask,co_attention_mask)  
                    # Calculate batch loss

                    loss = criterion(outputs,torch.unsqueeze(labels,1))
                # Add batch loss to list

                lossitem = loss.detach()
                if DISTRIBUTED:
                    dist.all_reduce(lossitem)
                    lossitem = lossitem.item()/n_gpu
                else:
                    lossitem = lossitem.item()
                running_loss_train += lossitem


                #print(f"Epoch({epoch}) -> batch {i}, loss: {loss.item()}, learning rate {warmupScheduler.get_last_lr()[0]}")
                
                ### Backward pass ###
                scaler.scale(loss).backward()       # Run backward pass with scaled graients
                scaler.step(optimizer)              # Run an optimizer step
                scale = scaler.get_scale()
                scaler.update()

                skip_lr_schedule = (scale > scaler.get_scale()) 
                            # # Append learning rate to list 
                learningRate.append(optimizer.param_groups[0]['lr'])
                
                if not skip_lr_schedule:
                    warmupScheduler.step() # update both lr schedulers 

                predY = (F.sigmoid(torch.squeeze(outputs)) > 0.5)
                groundT = (labels == 1)
                corrects = (torch.eq(predY,groundT).sum() / len(labels)).detach()
                accuracyitem = corrects
                accuracy_running += accuracyitem


                if is_main_process() or not DISTRIBUTED:
                    print(f"Epoch({epoch}) -> batch {i}, loss: {lossitem}, acc: {accuracyitem}, learning rate {optimizer.param_groups[0]['lr']}")

            # Calculate the avg loss of the training epoch and append it to list 
            epochLoss = running_loss_train/len(trainDL)
            trainingLoss.append(epochLoss)

            if DISTRIBUTED:
                dist.all_reduce(accuracy_running)
                accuracy_running= accuracy_running.item() /n_gpu
            else:
                accuracy_running = accuracy_running.item()
            epochAccuracy = accuracy_running/len(trainDL)
            accuracyTrain.append(epochAccuracy)

            if is_main_process() or not DISTRIBUTED:
                print(f"epoch loss: {epochLoss}")

            if DISTRIBUTED:
                    # wait for all gpus to get to here before saving
                dist.barrier()
            
            ########################################### Checkpoint every epoch ##########################################
            if is_main_process() or not DISTRIBUTED:
                if DISTRIBUTED:
                    model_checkpoint = model.module.state_dict()
                else:
                    model_checkpoint = model.state_dict()
                save_checkpoint(os.path.join(args.checkpoint_dir,str(epoch)),model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, epoch)

            ########################################### Validation  #####################################################

            model.eval() # Get model in eval mode

            running_loss_val = 0 # Keep track of avg loss for each epoch (val)
            accuracy_running_val = 0 
            for i, batch in enumerate(valDL):

                # Data related stuff
                batch = [t.cuda(device=device, non_blocking=True) for t in batch]
                textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, labels = batch
                #labels = torch.argmax(target,1)
                labels = torch.tensor([t.type(torch.float32)for t in labels], device=device)

                # Forward pass
                with amp.autocast(): # Cast from f32 to f16 
                    outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask,co_attention_mask)
                    
                    # Calculate batch loss 
                    loss = criterion(outputs,torch.unsqueeze(labels,1))

                lossitem = loss.detach()
                if DISTRIBUTED:
                    dist.all_reduce(lossitem)
                    lossitem = lossitem.item()/n_gpu
                else:
                    lossitem = lossitem.item()
                running_loss_val += lossitem


                predY = (torch.nn.functional.sigmoid(torch.squeeze(outputs)) > 0.5)
                groundT = (labels == 1)
                correctsval = (torch.eq(predY,groundT).sum() / len(labels)).detach()
                accuracyitem = correctsval
                accuracy_running_val +=accuracyitem


            # Calculate the avg loss of the validation epoch and append it to list 
            epochLossVal = running_loss_val/len(valDL)
            valLoss.append(epochLossVal)

            if DISTRIBUTED:
                dist.all_reduce(accuracy_running_val)
                accuracy_running_val= accuracy_running_val.item() /n_gpu
            else:
                accuracy_running_val = accuracy_running_val.item()
            accuracyEpochVal = accuracy_running_val/len(valDL)
            valAccuracy.append(accuracyEpochVal)

            if is_main_process() or not DISTRIBUTED:
                print(f"validation accuracy: {accuracyEpochVal}")

            reducelrScheduler.step(metrics=epochLossVal) # keep track of validation loss to reduce lr when necessary 
            
            # Update the progress bar 
            if is_main_process() or not DISTRIBUTED:
                pbarTrain.set_description(f"epoch: {epoch} / training loss: {round(epochLoss,3)} / lr: {warmupScheduler.get_last_lr()[0]}")
    except Exception as e:
        traceback.print_exc()
        # when sigterm caught, save checkpoint and exit
        
        # when sigterm caught, save checkpoint and exit
        if DISTRIBUTED:
                # wait for all gpus to get to here before saving
            dist.barrier()

        if is_main_process() or not DISTRIBUTED:
            # Check for dataparallel, the model state dictionary changes if wrapped arround nn.dataparallel
            if DISTRIBUTED:
                model_checkpoint = model.module.state_dict()
            else:
                model_checkpoint = model.state_dict()
            save_checkpoint(args.checkpoint_dir,model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, epoch)
        sys.exit(0)


    

    ############################################ Save completed checkpoint to the out folder #############################################
    if is_main_process() or not DISTRIBUTED :
        print("--- Training Complete, Saving checkpoint ---")
        if DISTRIBUTED:
            model_checkpoint = model.module.state_dict()
        else:
            model_checkpoint = model.state_dict()
        save_checkpoint(args.output_dir,model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, args.num_epochs, final = True)
    
    
    ####### Create plots and save them ######
    if is_main_process() or not DISTRIBUTED:
        plt.style.use(['seaborn']) # change style (looks cooler!)
        # learning rate 
        plt.figure(1)
        plt.plot(learningRate, "m-o", label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.savefig(f"{args.output_dir}/learningRate.png")


        # Train loss 
        plt.figure(3)
        plt.plot(trainingLoss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{args.output_dir}/TrainLoss.png")

        # Also save the lists and stats in a csv to create other plots if needed 
        with open(f"{args.output_dir}/ViLBERTlists.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["Learning rate over the epochs:"])
            write.writerow(learningRate)
            write.writerow(["Training loss over the epochs:"])
            write.writerow(trainingLoss)
            write.writerow(["Validation loss over the epochs:"])
            write.writerow(valLoss)
            write.writerow(["Training Acc over the epochs:"])
            write.writerow(accuracyTrain)
            write.writerow(["Validation acc over the epochs:"])
            write.writerow(valAccuracy)
            write.writerow(["--- stats ---"])
            write.writerow([f"Final training loss achieved {trainingLoss[len(trainingLoss)-1]}"])
            write.writerow([f"Final validation loss achieved {valLoss[len(valLoss)-1]}"])
        
if __name__=="__main__":
    main()
