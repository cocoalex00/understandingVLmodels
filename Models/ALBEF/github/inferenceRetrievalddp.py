# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ALBEF model to image classification.

# imports 
from selectors import EpollSelector
import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.use("pdf")
import traceback 
import pandas as pd

import csv
import signal
import sys 
import os
import torch.nn as nn
import numpy as np
import random 
from tqdm import tqdm
import argparse
import yaml
import utils
from dataset import create_sampler,create_loader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from distribuido import setup_for_distributed, save_on_master, is_main_process



from torch.utils.data import DataLoader
from transformers import   BertTokenizer
from dataset.places_dataset import places365
from models.model_imgclf import ALBEF
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

try:
    import torch.cuda.amp as amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

gpu_list = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

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
        type=str,
        #default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ALBEF/github/pretrained/ALBEF.pth",
        default= "/vol/teaching/HernandezDiazProject/understandingVLmodels/Models/ALBEF/github/pretrained/ALBEF.pth",
        #default = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ALBEF/github/pretrained/ALBEF.pth",
        help="Path to the pth file that contains the model's checkpoint"
    )
    parser.add_argument(
        "--local_rank",
        type=int
    )
    parser.add_argument(
        "--config",
        type=str,
        #default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/ALBEF/github/configs/Imgclf_places.yaml",
        default= "/vol/teaching/HernandezDiazProject/understandingVLmodels/Models/ALBEF/github/configs/Imgclf_places.yaml",
        help="Path to the config file for the task and model"
    )
    parser.add_argument(
        "--annotTest",
        type=str,
        default="/vol/teaching/HernandezDiazProject/places365_test_10retrieval.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--root_test",
        type=str,
       # default="/mnt/c/Users/aleja/Desktop/MSc Project/images/val_256/",
        default="/vol/teaching/HernandezDiazProject/Data/Places365/trainshmol/data_256/",
        help="Path to the images of the dataset (train)"
    )
    parser.add_argument(
        "--num_labels",
        default=1,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/vol/teaching/HernandezDiazProject/understandingVLmodels/Experiments/ALBEF/ret/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=365,
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
        default="/vol/teaching/HernandezDiazProject/retrieval_labels.txt",
        help="random seed for initialisation in multiple GPUs"
    )

    # Get all arguments 
    args = parser.parse_args()

    # load the config file
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

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

    n_gpu = torch.cuda.device_count()                                       # Check the number of GPUs available

    if is_main_process() or not DISTRIBUTED:
        print("the device being used is: " + str(device))
        print("number of gpus available: " + str(n_gpu))
        print(f"Using mixed precision training: {APEX_AVAILABLE}")

    # if n_gpu > 1: 
    #     args.batch_size = args.batch_size * n_gpu   # Augment batch size if more than one gpu is available

    
    #################################################### Dataset ##################################################################
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the dataset ##")

    TestDataset = places365(args.annotTest, args.root_test)


    if DISTRIBUTED:
        testSampler = DistributedSampler(dataset=TestDataset, shuffle=False)   
       

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
    model = ALBEF(config=config, text_encoder="bert-base-uncased", num_labels = args.num_labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 

    labels_to_text = pd.read_csv(args.labels_path, header=None, delimiter = "/")[0].values.tolist()

    ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.pretrained,"ALBEFFineTuned.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.pretrained,"ALBEFFineTuned.pth"))

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
            #sys.exit(1)



#    model.requires_grad_(False)

    
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




    ######################################################### Training loop ############################################################

    
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

                text = [category for _ in range (args.batch_size)]
                print(text)

                text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
                

                groundTruth = [index]* args.batch_size 
                print(groundTruth)

                predicted_category = []
                gt_category = []


                for i, batch in enumerate(testDl):
                    # Data related stuff
                    image, _, label = batch
                    image = image.to(device)
  
                    labels = [int(sape) for sape in np.equal(groundTruth,label.cpu())]
                    print(labels)
                    labels_tensor = torch.tensor(labels, dtype=torch.float32,device=device)


                     # Tokenize sentence


                    ### Forward pass ###
                    with amp.autocast(): # Cast from f32 to f16 
                        output = model(image,text_inputs)

                    #labels = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


                    predicted_batch = torch.nn.functional.sigmoid(torch.squeeze(output))
                    print(predicted_batch)
                    predicted_batch = predicted_batch.tolist()

                    # Add predictions and groundtruth to their corresponding lists
                    predicted_category += predicted_batch
                    gt_category += labels


                if DISTRIBUTED:
                    world_size = int(os.environ["WORLD_SIZE"])

                    #turn arrays into tensors so that all_gather works
                    predicted_category = torch.tensor(predicted_category,device=device)
                    print(f"gpu : {dist.get_rank} -> {predicted_category}")
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
        with open(f"{args.output_dir}/ALBEFretInference.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["p@1","p@5","p@10"])
            write.writerow([avgp1,avgp5,avgp10])
            write.writerow(["r@1","r@5","r@10"])
            write.writerow([avgrec1,avgrec5,avgrec10])
        
if __name__=="__main__":
    main()
