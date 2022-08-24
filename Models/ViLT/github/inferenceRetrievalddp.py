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
import pandas as pd

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


def process_metrics(fullGT, fullPreds):
    '''
    This function calculates the final retrieval metrics to evaluate the systems 
    fullGT: list containing the ground truth for all the elements in 
    '''
    precisionAt1 = []
    recallAt1 = []
    precisionAt5 = []
    recallAt5 = []
    precisionAt10 = []
    recallAt10 = []
    for gtCat, predCat in zip(fullGT, fullPreds):
        #Get topk predictions and their indices 
        topkPreds, indices = torch.topk(torch.tensor(predCat),10)

        print(indices)
        #Get ground truth for the elements of the predicted indices
        gtretrieved = np.array(gtCat)[indices.tolist()]
        print(gtretrieved)
        
        # get true positives of top 1, 5 and 10 results
        tp1 = np.sum(gtretrieved[0])
        print(tp1)

        tp5 = np.sum(gtretrieved[:5])

        tp10 = np.sum(gtretrieved)
        print(tp10)

        # Append precision and recal
        precisionAt1.append(tp1/1)
        precisionAt5.append(tp5/5)
        precisionAt10.append(tp10/10)

        # Append precision and recal
        recallAt1.append(tp1/10)
        recallAt5.append(tp5/10)
        recallAt10.append(tp10/10)
    
    return [precisionAt1, precisionAt5, precisionAt10], [recallAt1,recallAt5,recallAt10]



MAXSEQUENCELENGTH = 7
def create_text_batch(text: str, tokenizer, mlm_collator, batch_size= 32):

    
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAXSEQUENCELENGTH,
        return_special_tokens_mask=True,
    )
    print(text)
    print(encoding)

    text_ids = [encoding["input_ids"] for _ in range(batch_size)]
    text_labels_mlm = [encoding["input_ids"] for _ in range(batch_size)]
    text_labels = [[-100 for _ in range(MAXSEQUENCELENGTH)] for _ in range(batch_size)]
    attention_mask = [encoding["attention_mask"] for _ in range(batch_size)]


    
    #dict_batch = [dict for _ in range(batch_size)]

    # print(dict_batch["text"][0])

    # txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    #text = 

    # if len(txt_keys) != 0:
    #     texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
    #     encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
    #     draw_text_len = len(encodings)
    #     flatten_encodings = [e for encoding in encodings for e in encoding]
    #     flatten_mlms = mlm_collator(flatten_encodings)

    #     for i, txt_key in enumerate(txt_keys):
    #         texts, encodings = (
    #             [d[0] for d in dict_batch[txt_key]],
    #             [d[1] for d in dict_batch[txt_key]],
    #         )

    #         mlm_ids, mlm_labels = (
    #             flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
    #             flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
    #         )

    #         input_ids = torch.zeros_like(mlm_ids)
    #         attention_mask = torch.zeros_like(mlm_ids)
    #         for _i, encoding in enumerate(encodings):
    #             _input_ids, _attention_mask = (
    #                 torch.tensor(encoding["input_ids"]),
    #                 torch.tensor(encoding["attention_mask"]),
    #             )
    #             input_ids[_i, : len(_input_ids)] = _input_ids
    #             attention_mask[_i, : len(_attention_mask)] = _attention_mask

        # dict_batch[txt_key] = texts
        # dict_batch[f"{txt_key}_ids"] = input_ids
        # dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
        # dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
        # dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
        # dict_batch[f"{txt_key}_masks"] = attention_mask

    return text_ids, text_labels_mlm, text_labels, attention_mask


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
        #default="/vol/teaching/HernandezDiazProject/understandingVLmodels/Models/ViLT/github/pretrained/vilt_200k_mlm_itm.ckpt",
        default = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ViLT/github/pretrained/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="Path to the checkpoint file containing the model's weights and stuff",
    )
    parser.add_argument(
        "--data_path",
        #default="/vol/teaching/HernandezDiazProject/Data/arrowfiles",
        default = "/mnt/c/Users/aleja/Desktop/MSc Project/totest",
        type=str,
        help="Path to the folder where the dataset file (.arrow) lives",
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
        #default="/vol/teaching/HernandezDiazProject/understandingVLmodels/Experiments/ViLT/imgClf/outfull",
        default ="/mnt/c/Users/aleja/Desktop/MSc Project/",
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
    parser.add_argument(
        "--labels_path",
        type=str,
       # default="/vol/teaching/HernandezDiazProject/retrieval_labels.txt",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/retrieval_labels.txt",
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

    TestDataset = Places365(args.data_path, split="val")   ####### Change this to train for the HTCONDOR!!!!!!!

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
    model = ViLTransformerSS(_config,"placesret")

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
 
    labels_to_text = pd.read_csv(args.labels_path, header=None, delimiter = "/")[0].values.tolist()


    print("***** Running inference *****")
    print("  Batch size: ", args.batch_size)


    
    try:
        with torch.no_grad():
        ########################################### Validation  #####################################################

            model.eval() # Get model in eval mode

            running_loss_val = 0 # Keep track of avg loss for each epoch (val)
            predictedFull = [] # Keep track of avg loss for each epoch (val)
            groundTruthFull = []
            for index, category in enumerate(labels_to_text[:3]):

                #text = [category for _ in range (args.batch_size)]

                text_ids, text_labels_mlm, text_labels, attention_mask = create_text_batch(category,tokenizer,collator,args.batch_size)

                text_tensor = torch.tensor(text_ids).to(device)
                text_labels_mlm_tensor = torch.tensor(text_labels_mlm).to(device)
                text_labels_tensor = torch.tensor(text_labels).to(device)
                attention_mask_tensor = torch.tensor(attention_mask).to(device)


                print(text_tensor)
                ##text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
                

                groundTruth = [index]* args.batch_size 
                print(groundTruth)

                predicted_category = []
                gt_category = []


                for i, batch in enumerate(testDl):

                    #print(batch)
                    # Data related stuff
                    #print(batch["label"])
                    label = torch.tensor(batch["label"]).squeeze().tolist()
                    #print(label)
                    labels = [int(sape) for sape in np.equal(groundTruth,label)]
                    #print(labels)
                    batch["text_ids"] = text_tensor
                    batch["text_labels"] = text_labels_tensor
                    batch["text_masks"] = attention_mask_tensor
                    batch["text_labels_mlm"] = text_labels_mlm_tensor

                    

                    # label = torch.tensor(batch["label"]).to(device).squeeze()
                    # for i in batch.keys():
                    #     if type(batch[i]) != list:
                    #         batch[i] = batch[i].to(device)

                    batch["image"] = [batch["image"][0].to(device)]
                
                    print(batch)
                    ### Forward pass ###
                    with amp.autocast(): # Cast from f32 to f16 
                        output = model(batch)

                    #labels = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


                    predicted_batch = torch.nn.functional.sigmoid(torch.squeeze(output["imgcls_logits"]))

                    predicted_batch = predicted_batch.tolist()

                    # Add predictions and groundtruth to their corresponding lists
                    predicted_category += predicted_batch
                    gt_category += labels


                if DISTRIBUTED:
                    world_size = int(os.environ["WORLD_SIZE"])

                    #turn arrays into tensors so that all_gather works
                    predicted_category = torch.tensor(predicted_category,device=device)

                    gt_category = torch.tensor(gt_category,device=device)
                    #create the containers
                    predicted_container = [torch.zeros_like(predicted_category) for _ in range(2)]
                    gt_container = [torch.zeros_like(gt_category) for _ in range(2)]

                    #run gather
                    dist.all_gather(predicted_container,predicted_category)
                    dist.all_gather(gt_container,gt_category)
                    #concatenate lists and add them to full lists of categories
                    predictedFull.append(torch.cat(predicted_container).tolist())
                    groundTruthFull.append(torch.cat(gt_container).tolist())
                else:
                    predictedFull.append(predicted_category)
                    groundTruthFull.append(gt_category)
                    
                print(f"Done: {category}")

            print(groundTruthFull)
            print(predictedFull)

            precisionFinal , recallFinal = process_metrics(groundTruthFull,predictedFull)

            # get average precision and recall 
            avgp1 = sum(precisionFinal[0])/len(precisionFinal[0])
            avgp5 = sum(precisionFinal[1])/len(precisionFinal[1])
            avgp10 = sum(precisionFinal[2])/len(precisionFinal[2])

            avgrec1 = sum(recallFinal[0])/len(recallFinal[0])
            avgrec5 = sum(recallFinal[1])/len(recallFinal[1])
            avgrec10 = sum(recallFinal[2])/len(recallFinal[2])
 
            if is_main_process() or not DISTRIBUTED:
                print(f"P@1: {avgp1} / P@5: {avgp5} / P@10: {avgp10}")
                print(f"R@1: {avgrec1} / R@5: {avgrec5} / R@10: {avgrec10}")
           
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
            write.writerow(["p@1","p@5","p@10"])
            write.writerow([avgp1,avgp5,avgp10])
            write.writerow(["r@1","r@5","r@10"])
            write.writerow([avgrec1,avgrec5,avgrec10])
        

