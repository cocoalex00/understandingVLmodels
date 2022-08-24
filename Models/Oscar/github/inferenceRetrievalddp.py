# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the training script to fine-tune the adapted ViLBERT model to image classification.

# imports 
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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from distribuido import setup_for_distributed, save_on_master, is_main_process
import traceback

from oscar.run_imageclassif import Places365, load_obj_tsv
from torch.utils.data import DataLoader
from transformers import  get_linear_schedule_with_warmup
from pytorch_transformers import BertConfig, BertTokenizer
import pandas as pd

from oscar.modeling.modeling_bert import OscarForImageClassification
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
    if final:
        torch.save(obj=checkpoint, f=f"{folder_path}/OscarFineTuned.pth")
    else:
        torch.save(obj=checkpoint, f=f"{folder_path}/checkpointOscar.pth")
    print("--Checkpoint saved--")





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

MAXSEQUENCELENGTH = 7
def create_text_batch(text: str, tokenizer, batch_size= 32):


    
    # tokenize sentence
    tokens= tokenizer.tokenize(text)   # Tokens are only [CLS] [SEP]

    # Check if larger than maxlength
    if len(tokens) > MAXSEQUENCELENGTH - 2:
            tokens = tokens[:(MAXSEQUENCELENGTH- 2)]

    # Add separator tokens to tokens and create the segment ids
    tokens = tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    # add class token to the tokens string and to the segment ids
    tokens = ["CLS"] + tokens
    segment_ids = [0] + segment_ids

    # Convert the tokenized sentence to ids 
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)


    # No zero padding to the right 
    padding_length = MAXSEQUENCELENGTH- len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    input_mask = input_mask + [1] * 36

    padding_matrix = torch.zeros((2*30 - 36, 2048))
    input_mask = input_mask + ([0] * padding_matrix.shape[0])

    # text = " "
    # tokens = tokenizer.tokenize(text)
    # tokens = tokens[: MAXSEQUENCELENGTH -2]

    # tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # tokens = tokenizer.convert_tokens_to_ids(tokens)

    # #print(tokens)


    # segment_ids = [0] * len(tokens)
    # input_mask = [1] * len(tokens)
    # if len(tokens) < MAXSEQUENCELENGTH:
    #     # Note here we pad in front of the sentence
    #     padding = [0] * (MAXSEQUENCELENGTH - len(tokens))
    #     tokens = tokens + padding
    #     input_mask += padding
    #     segment_ids += padding
    # input_mask = input_mask + [1] * 36
    # input_mask = input_mask + [0] * (2*30 - 36)
    # #print(torch.tensor(input_mask).shape)

    tokens_batch = np.tile(input_ids,(batch_size,1))
    input_mask_batch = np.tile(input_mask,(batch_size,1))
    segment_ids_batch = np.tile(segment_ids,(batch_size,1))

    return tokens_batch, input_mask_batch, segment_ids_batch

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

        #Get ground truth for the elements of the predicted indices
        gtretrieved = np.array(gtCat)[indices.tolist()]
        
        # get true positives of top 1, 5 and 10 results
        tp1 = np.sum(gtretrieved[0])
        tp5 = np.sum(gtretrieved[:5])
        tp10 = np.sum(gtretrieved)

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

    # ARGUMENTS NEEDED FOR THE MODEL
    parser.add_argument("--data_dir", default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/", type=str,
                        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: ")
    parser.add_argument("--pretrained", default="pretrained/base-vg-labels/ep_107_1192087", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default="places", type=str, 
                        help="The name of the task to train selected in the list: ")

    parser.add_argument("--data_label_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--train_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or all")
    parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")
    parser.add_argument("--use_layernorm", action='store_true', help="use_layernorm")
    parser.add_argument("--use_label_seq", action='store_true', help="use_label_seq")
    parser.add_argument("--use_pair", action='store_true', help="use_pair")
    parser.add_argument("--num_choice", default=2, type=int, help="num_choice")
    parser.add_argument("--max_seq_length", default=7, type=int, help="max lenght of text sequence")
    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int)
    parser.add_argument("--code_voc", default=512, type=int)
    parser.add_argument("--img_feature_type", default="faster_r-cnn", type=str)
    parser.add_argument("--classifier", default="mlp", type=str)
    parser.add_argument("--cls_hidden_scale", default=2, type=int)

    ####
    parser.add_argument(
        "--tsv_test",
        type=str,
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/val/",
        help="Path to the tsv file containing the features of the dataset (train)"
    )
    parser.add_argument(
        "--num_labels",
        default=1,
        type=int,
        help="Number of classes in the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/Oscar/ret/out",
        type=str,
        help="The output directory where the fine-tuned model and final plots will be saved.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Experiments/Oscar/ret/checkpoints",
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
        "--seed",
        type=int,
        default=0,
        help="random seed for initialisation in multiple GPUs"
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        #default="/vol/teaching/HernandezDiazProject/retrieval_labels.txt",
        default="/mnt/c/Users/aleja/Desktop/MSc Project/totest/retrieval_labels.txt",
        help="random seed for initialisation in multiple GPUs"
    )

    # Get all arguments 
    args = parser.parse_args()


    # The oscar dataset needs for the features to be loaded and the tokenizer to be initialised
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=args.num_labels, finetuning_task=args.task_name)

    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.use_layernorm = args.use_layernorm
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    config.num_choice = args.num_choice

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


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

    # featuresTrain = load_obj_tsv(args.tsv_train)
    # featuresVal = load_obj_tsv(args.tsv_val)
    testDataset = Places365(args,"test",args.tsv_test,tokenizer) 
    

    if DISTRIBUTED:
        testSampler = DistributedSampler(dataset=testDataset, shuffle=True)   
       

        testDl = DataLoader(
            dataset= testDataset,
            batch_size= args.batch_size,
            #shuffle= True,
            pin_memory=True,
            sampler=testSampler
        )
    else:
        testDl = DataLoader(
            dataset= testDataset,
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
    model = OscarForImageClassification(config=config)
      
      ##################################################### Checkpoint or pre-trained ###################################################
    if os.path.exists(os.path.join(args.pretrained,"OscarFineTuned.pth")):
        if is_main_process() or not DISTRIBUTED:
            print(f"Checkpoint found, loading")

        checkpoint = torch.load(os.path.join(args.pretrained,"OscarFineTuned.pth"))

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

    # Loss fnc and optimizer
    criterion = CrossEntropyLoss()
    
    model.requires_grad_(False)



    # Lists to keep track of nice stuff like loss and lr 
    testLoss = []
    accuracyTest = []



    # # Check for available checkpoints 
    # if os.path.exists(os.path.join(args.checkpoint_dir,"checkpointOscar.pth")):
    #     print(f"Checkpoint found, loading")
    #     checkpoint = torch.load(os.path.join(args.checkpoint_dir,"checkpointOscar.pth"))
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
    
   


    # Progress bars

    # Training loop ####################################################################################################################################

    print("***** Running inference *****")
    print("  Batch size: ", args.batch_size)

    labels_to_text = pd.read_csv(args.labels_path, header=None, delimiter = "/")[0].values.tolist()


    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


    try:
        with torch.no_grad():
        ########################################### Validation  #####################################################

            model.eval() # Get model in eval mode

            running_loss_val = 0 # Keep track of avg loss for each epoch (val)
            predictedFull = [] # Keep track of avg loss for each epoch (val)
            groundTruthFull = []
            for index, category in enumerate(labels_to_text[:3]):

                text, input_mask, segment_ids = create_text_batch(category, tokenizer = tokenizer, batch_size= args.batch_size)
                
                text_tensor = torch.tensor(text, device=device, dtype=torch.long) 
                segid_tensor = torch.tensor(segment_ids, device=device, dtype=torch.long)
                inputmask_tensor = torch.tensor(input_mask, device=device, dtype=torch.long)




                groundTruth = [index]* args.batch_size 

                predicted_category = []
                gt_category = []


                for i, batch in enumerate(testDl):
                    
                    batch = [t.cuda(device=device, non_blocking=True) for t in batch]
                    _, _, _, labels, img_feats= batch


                    labels = [int(sape) for sape in np.equal(groundTruth,labels.cpu())]

                    #print(mask.shape)

                    #print(mask)

                    ### Forward pass ###
                    with amp.autocast(): # Cast from f32 to f16 
                        output = model(input_ids = text_tensor,attention_mask=  inputmask_tensor, position_ids = segid_tensor, img_feats= img_feats)

                    


                    predicted_batch = torch.nn.functional.sigmoid(torch.squeeze(output[0]))
                    #print(predicted_batch)
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
                    gt_container = predicted_container

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
        with open(f"{args.output_dir}/OscarInference.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(["p@1","p@5","p@10"])
            write.writerow([avgp1,avgp5,avgp10])
            write.writerow(["r@1","r@5","r@10"])
            write.writerow([avgrec1,avgrec5,avgrec10])
        
if __name__=="__main__":
    main()
