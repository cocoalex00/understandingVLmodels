# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted LXMERT model to image classification.

# imports 
import enum
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import torch
import pandas as pd

from src.tasks.places_data import PlacesDataset, PlacesTorchDataset
from torch.utils.data import DataLoader
from src.tasks.retrieval_model import retrievalModel
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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from distribuido import setup_for_distributed, save_on_master, is_main_process


from torch.optim.lr_scheduler import (
    ReduceLROnPlateau
)

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
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/LXMERT//out50",
        type=str,
        help="PATH to the .pth file contatining the pre-trained weights. Ojo, the function loads it like 'Path + _LXRT.pth' so omit that part"
    )
    #### Annotations and TSV files 
    parser.add_argument(
        "--annotTest",
        type=str,
        #default="/mnt/fast/nobackup/scratch4weeks/ah02299/Utilities/jsonfiles/places365_test_alexsplit.json",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/totest.json",
        help="Path to the jsonline file containing the annotations of the dataset"
    )
    parser.add_argument(
        "--local_rank",
        type=int
    )

    parser.add_argument(
        "--tsv_test",
        type=str,
        #default="/mnt/fast/nobackup/scratch4weeks/ah02299/train/",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (train)"
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
        default="/mnt/fast/nobackup/scratch4weeks/ah02299/understandingVLmodels/Experiments/LXMERT/ret/out50",
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
        testsampler = DistributedSampler(dataset=TestDataset, shuffle=False)   

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
            shuffle= False,
            pin_memory=True,
        )

    if is_main_process()  or not DISTRIBUTED:
        print("## Dataset loaded succesfully ##")


    #################################################### MODEL STUFF ################################################################
    # Load the Model
    if is_main_process() or not DISTRIBUTED:
        print("## Loading the Model ##")
    # Load the model and freeze everything up to the last linear layers (Image classifier)
    model = retrievalModel(args.num_labels)

    print(model)


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


   
    


    # Lists to keep track of nice stuff like loss and lr 
    testLoss = []
    accuracyTest = []

    labels_to_text = pd.read_csv(args.labels_path, header=None, delimiter = "/")[0].values.tolist()


        ##################################################### Checkpoint or pre-trained ###################################################


    
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

                txt = [category for _ in range (args.batch_size)]


                

                groundTruth = [index]* args.batch_size 
                print(groundTruth)

                predicted_category = []
                gt_category = []


                for i, batch in enumerate(testDl):
                    # Data related stuff

                    _, imgfeats, boxes, labels, objectsid, img_id = batch
                    labels = [int(sape) for sape in np.equal(groundTruth,labels.cpu())]
                    #print(labels)
                
                    
                    imgfeats, boxes=  imgfeats.cuda(), boxes.cuda()



                    ### Forward pass ###
                    with amp.autocast(): # Cast from f32 to f16 
                        outputs = model(imgfeats,boxes,txt)


                    #labels = torch.tensor([t.type(torch.LongTensor)for t in labels], device=device)


                    predicted_batch = torch.nn.functional.sigmoid(torch.squeeze(outputs))

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
                    predicted_container = [torch.zeros_like(predicted_category) for _ in range(world_size)]
                    gt_container = [torch.zeros_like(gt_category) for _ in range(world_size)]

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
        with open(f"{args.output_dir}/LXMERTInference.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["inference loss:"])
            write.writerow(testLoss)
            write.writerow(["inference Accuracy"])
            write.writerow(accuracy)
if __name__=="__main__":
    main()
