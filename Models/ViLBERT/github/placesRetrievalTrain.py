# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ViLBERT model to image classification.

# imports 
from typing import Iterable
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
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

from vilbert.vilbertRetrievalPlaces import VILBertForRetrieval
from vilbert.datasets.ITretrievalDataset import get_data_loader 


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
        "--local_rank",
        type=int,
    )
    parser.add_argument(
        "--from_pretrained",
        #default= "bert-base-uncased",
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Models/ViLBERT/github/save/pretrained_model.bin",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
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
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (validation)"
    )
    ####
    parser.add_argument(
        "--num_labels",
        default=1,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/ViLBERT/ret/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/ViLBERT/ret/checkpoints",
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
        help="The base learning rate to be used with the optimizer (default =3e-4)"
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
    n_gpu = torch.cuda.device_count()                                       # Check the number of GPUs available


    print("the device being used is: " + str(device))
    print("number of gpus available: " + str(n_gpu))
    print(f"Using mixed precision training: {APEX_AVAILABLE}")

    if n_gpu > 1: 
        args.batch_size = args.batch_size * n_gpu   # Augment batch size if more than one gpu is available

    
    # Load the dataset
    print("## Loading the dataset ##")

    trainDL = get_data_loader(args.annotTrain, args.labels_path,  args.tsv_train,  args.batch_size)
    valDL = get_data_loader(args.annotVal, args.labels_path, args.tsv_val, args.batch_size)


    print("## Dataset loaded succesfully ##")


    ##### MODEL STUFF #####
    # Load the Model
    print("## Loading the Model ##")
    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = VILBertForRetrieval(args.config_file, args.from_pretrained)
    # Only train the last classifier
    model.vilbertBase.requires_grad_(False)

    

    print("## Model Loaded ##")

    # Data paralellization in multiple gpus if more than one is available (only in one node, multiple is too hard f***)
    if n_gpu > 1:
        device = torch.device("cuda")
        model = nn.DataParallel(model)
        model.to(device)
        print("Data parallelization activated")
    else:
        model.to(device)

    # Loss fnc and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), args.lr)
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


    # Check for available checkpoints 
    if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth")):
        print(f"Checkpoint found, loading")
        checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointVilbert.pth"))
        start_epoch = checkpoint['current_epoch']
        learningRate = checkpoint['lr']
        trainingLoss = checkpoint['Tloss']
        valLoss = checkpoint['Vloss']

        # Check if model has been wrapped with nn.DataParallel. This makes loading the checkpoint a bit different
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_checkpoint'])
        else:
            model.load_state_dict(checkpoint['model_checkpoint'])
        optimizer.load_state_dict(checkpoint['optimizer_checkpoint'])
        warmupScheduler.load_state_dict(checkpoint['warmup_checkpoint'])
        reducelrScheduler.load_state_dict(checkpoint['scheduler_checkpoint'])


        print("Checkpoint loaded")
    else:
        print("No checkpoint found, starting fine tunning from base pre-trained model")


    # Progress bars
    pbarTrain = tqdm(range(start_epoch, args.num_epochs))

    # Training loop ####################################################################################################################################

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
                batch = [t.cuda(device=device, non_blocking=True) for t in batch]
                textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, labels = batch
                #labels = torch.argmax(target,1)
                labels = torch.tensor([t.type(torch.float32)for t in labels], device=device)

                ### Forward pass ###
                with amp.autocast(): # Cast from f32 to f16 
                    outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask)
                    # Calculate batch loss
 
                    loss = criterion(torch.squeeze(outputs),labels)
                # Add loss to list
                running_loss_train += loss.item()

                
                ### Backward pass ###
                scaler.scale(loss).backward()       # Run backward pass with scaled graients
                scaler.step(optimizer)  
                scale = scaler.get_scale()            # Run an optimizer step
                scaler.update()


                skip_lr_schedule = (scale > scaler.get_scale()) 
                            # # Append learning rate to list 
                learningRate.append(optimizer.param_groups[0]['lr'])
                
                if not skip_lr_schedule:
                    warmupScheduler.step() # update both lr schedulers 
                print(f"Epoch({epoch}) -> batch {i}, loss: {loss.item()}, learning rate {optimizer.param_groups[0]['lr']}")

            # Calculate the avg loss of the training epoch and append it to list 
            epochLoss = running_loss_train/len(trainDL)
            trainingLoss.append(epochLoss)
            
             # Save frequent checkpoints
            if isinstance(model,nn.DataParallel):
                model_checkpoint = model.module.state_dict()
            else:
                model_checkpoint = model.state_dict()
            save_checkpoint(os.path.join(args.checkpoint_dir,str(epoch)),model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, epoch)



            ########################################### Validation  #####################################################

            model.eval() # Get model in eval mode

            running_loss_val = 0 # Keep track of avg loss for each epoch (val)
            for i, batch in enumerate(valDL):

                # Data related stuff
                batch = [t.cuda(device=device, non_blocking=True) for t in batch]
                textInput, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, labels = batch
                #labels = torch.argmax(target,1)
                
                labels = torch.tensor([t.type(torch.float32)for t in labels], device=device)
                

                # Forward pass
                with amp.autocast(): # Cast from f32 to f16 
                    outputs, no, _, _, _, _, _, _, _, _, _, _, _, = model(textInput, features, spatials, segment_ids, input_mask, image_mask)

                    # Calculate batch loss 
                    loss = criterion(outputs,labels.unsqueeze(1))
                # Add loss to list (val)
                running_loss_val += loss.item()

            # Calculate the avg loss of the validation epoch and append it to list 
            epochLossVal = running_loss_val/len(valDL)
            valLoss.append(epochLossVal)

            reducelrScheduler.step(metrics=epochLossVal) # keep track of validation loss to reduce lr when necessary 

            # # Update the progress bar 
            pbarTrain.set_description(f"epoch: {epoch} / training loss: {round(epochLoss,3)} validation loss: {round(epochLossVal,3)}")
    except Exception as e:
        print(e)
        # when sigterm caught, save checkpoint and exit
        
        # Check for dataparallel, the model state dictionary changes if wrapped arround nn.dataparallel
        if isinstance(model,nn.DataParallel):
            model_checkpoint = model.module.state_dict()
        else:
            model_checkpoint = model.state_dict()
        save_checkpoint(args.checkpoint_dir,model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, epoch)
        sys.exit(0)

    print("--- Training Complete, Saving checkpoint ---")

    plt.figure(1)
    plt.plot(learningRate, "m-o", label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.savefig(f"{args.output_dir}/learningRate.png")
    ####### Save completed checkpoint to the out folder ######
    if isinstance(model,nn.DataParallel):
        model_checkpoint = model.module.state_dict()
    else:
        model_checkpoint = model.state_dict()
    save_checkpoint(args.output_dir,model_checkpoint,optimizer.state_dict(),warmupScheduler.state_dict(),reducelrScheduler.state_dict(),trainingLoss,valLoss,learningRate, args.num_epochs, final = True)
    
    
    ####### Create plots and save them ######
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
    with open(f"{args.output_dir}/ViLBERTlists.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(["Learning rate over the training steps:"])
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
