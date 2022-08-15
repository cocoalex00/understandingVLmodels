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
        "--from_pretrained",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/snap/pretrained/model",
        type=str,
        help="PATH to the .pth file contatining the pre-trained weights. Ojo, the function loads it like 'Path + _LXRT.pth' so omit that part"
    )
    #### Annotations and TSV files 
    parser.add_argument(
        "--annotTrain",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/totest.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--local_rank",
        type=int
    )
    parser.add_argument(
        "--annotVal",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/totest.json",
        help="Path to the json file containing the annotations of the dataset (validation)"
    )
    parser.add_argument(
        "--tsv_train",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (train)"
    )
    parser.add_argument(
        "--tsv_val",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (validation)"
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
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/LXMERT/imgClf/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/LXMERT/imgClf/checkpoints",
        type=str,
        help="The output directory where the training checkpoints will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The number of samples in each batch.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
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

    trainDataset = PlacesTorchDataset(PlacesDataset(args.annotTrain, args.tsv_train))
    valDataset = PlacesTorchDataset(PlacesDataset(args.annotVal, args.tsv_val))

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
    model = imgClassifModel(args.num_labels)
    model.lxrt_encoder.load(args.from_pretrained,device) # load encoder's weight
    # Only train the last classifier
    model.lxrt_encoder.requires_grad_(False)

    

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
    learningrate = np.sqrt(n_gpu) * args.lr
    optimizer = AdamW(model.parameters(), learningrate)
    # Create gradient scaler for f16 precision
    scaler = amp.GradScaler()

    # learning rate schedulers 
    reducelrScheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.2, patience=1, cooldown=1, threshold=0.001
        )

    totalSteps = len(trainDL) * args.num_epochs
    warmupSteps = int(totalSteps * args.warmup_proportion)                         # Calculate the number of epochs to warm up for
    warmupScheduler = get_linear_schedule_with_warmup(optimizer= optimizer,
                                                    num_warmup_steps= warmupSteps,
                                                    num_training_steps = totalSteps
                                                    )
    



    # Lists to keep track of nice stuff like loss and lr 
    trainingLoss = []
    valLoss = []
    learningRate = []
    start_epoch = 0


    # # Check for available checkpoints 
    # if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointLXMERT.pth")):
    #     print(f"Checkpoint found, loading")
    #     checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointLXMERT.pth"))
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

        ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointLXMERT.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointLXMERT.pth"), map_location=device)
        start_epoch = checkpoint['current_epoch']
        learningRate = checkpoint['lr']
        trainingLoss = checkpoint['Tloss']
        valLoss = checkpoint['Vloss']

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if DISTRIBUTED:
            model.module.load_state_dict(checkpoint['model_checkpoint'], strict= False, map_location=device)
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

            running_loss_train = 0 # Keep track of avg loss for each epoch (train)
            for i, batch in enumerate(trainDL):
                optimizer.zero_grad()               # Clear gradients of the optimizer
            

                # Data related stuff
                txt, imgfeats, boxes, labels, objectsid, img_id = batch
                imgfeats, boxes=  imgfeats.cuda(), boxes.cuda()

                labelsTensor = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


                ### Forward pass ###
                with amp.autocast(): # Cast from f32 to f16 
                    outputs = model(imgfeats,boxes,txt)

                    # Calculate batch loss
                    loss = criterion(outputs,labelsTensor)
                # Add batch loss to list
                lossitem = loss.detach()
                dist.all_reduce(lossitem)
                lossitem = lossitem.item()/n_gpu
                running_loss_train += lossitem

                
                
                ### Backward pass ###
                scaler.scale(loss).backward()       # Run backward pass with scaled graients
                scaler.step(optimizer)              # Run an optimizer step
                scale = scaler.get_scale()
                scaler.update()

                skip_lr_schedule = (scale > scaler.get_scale()) 
                # Append learning rate to list 
                learningRate.append(optimizer.param_groups[0]['lr'])
                
                if not skip_lr_schedule:
                    warmupScheduler.step() 
                
                if is_main_process() or not DISTRIBUTED:
                    print(f"Epoch({epoch}) -> batch {i}, loss: {lossitem}, learning rate {optimizer.param_groups[0]['lr']}")

            # Calculate the avg loss of the training epoch and append it to list 
            epochLoss = running_loss_train/len(trainDL)
            trainingLoss.append(epochLoss)
            	
            
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

            model.eval() # Get model in eval mode

            running_loss_val = 0 # Keep track of avg loss for each epoch (val)
            for i, batch in enumerate(valDL):

                # Data related stuff
                txt, imgfeats, boxes, labels, objectsid, img_id = batch
                imgfeats, boxes =  imgfeats.cuda(), boxes.cuda()

                labelsTensor = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


                ### Forward pass ###
                with amp.autocast(): # Cast from f32 to f16 
                    outputs = model(imgfeats,boxes,txt)
                
                    # Calculate batch loss
                    loss = criterion(outputs,labelsTensor)
                # Add loss to list (val)
                lossitem = loss.detach()
                dist.all_reduce(lossitem)
                lossitem = lossitem.item()/n_gpu
                running_loss_val += lossitem
                
            # Calculate the avg loss of the validation epoch and append it to list 
            epochLossVal = running_loss_val/len(valDL)
            valLoss.append(epochLossVal)


            reducelrScheduler.step(metrics=epochLossVal) # keep track of validation loss to reduce lr when necessary 
            
            
            # Update the progress bar 
            if is_main_process() or not DISTRIBUTED:
                pbarTrain.set_description(f"epoch: {epoch} / training loss: {round(epochLoss,3)} / validation loss: {round(epochLossVal,3)} / lr: {warmupScheduler.get_last_lr()[0]}")
    except Exception as e:
        # when sigterm caught, save checkpoint and exit
        print(e)
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
        
    
    ###################################################### Create plots and save them ########################################################
    if is_main_process() or not DISTRIBUTED:
        plt.style.use(['seaborn']) # change style (looks cooler!)
        # learning rate 
        plt.figure(1)
        plt.plot(learningRate, "m-o", label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.savefig(f"{args.output_dir}/learningRate.png")

        # Combined loss 
        plt.figure(2)
        plt.plot(trainingLoss,"r-o", label="Train Loss", )
        plt.plot(valLoss,"-p", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{args.output_dir}/CombinedLoss.png")

        # Train loss 
        plt.figure(3)
        plt.plot(trainingLoss, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{args.output_dir}/TrainLoss.png")

        # Val loss 
        plt.figure(4)
        plt.plot(valLoss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{args.output_dir}/ValLoss.png")

        # Also save the lists and stats in a csv to create other plots if needed 
        with open(f"{args.output_dir}/LXMERTlists.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["Learning rate over the epochs:"])
            write.writerow(learningRate)
            write.writerow(["Training loss over the epochs:"])
            write.writerow(trainingLoss)
            write.writerow(["Validation loss over the epochs:"])
            write.writerow(valLoss)
            write.writerow(["--- stats ---"])
            write.writerow([f"Final training loss achieved {trainingLoss[len(trainingLoss)-1]}"])
            write.writerow([f"Final validation loss achieved {valLoss[len(valLoss)-1]}"])
        
if __name__=="__main__":
    main()
