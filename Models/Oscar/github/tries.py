
from os import O_EXCL
import torch
from oscar.run_imageclassif import Places365, load_obj_tsv
from pytorch_transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from oscar.modeling.modeling_bert import OscarForImageClassification
from pytorch_transformers.optimization import AdamW
import torch.nn as nn 

from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_VQA_LENGTH = 2
NUM_CLASSES = 365
LR= 0.000002
NUM_EPOCHS= 300
BATCH_SIZE = 2
WEIGHT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/snap/pretrained/model"
DATAPATH = "nlvr2_img_frcnn_feats.pt"


import argparse

if __name__ == "__main__":
   parser = argparse.ArgumentParser()

    ## Required parameters
   parser.add_argument("--data_dir", default="/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/Oscar/github/Places", type=str,
                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
   parser.add_argument("--model_type", default="bert", type=str,
                     help="Model type selected in the list: ")
   parser.add_argument("--model_name_or_path", default="pretrained/base-vg-labels/ep_107_1192087", type=str,
                     help="Path to pre-trained model or shortcut name selected in the list: ")
   parser.add_argument("--task_name", default="places", type=str, 
                     help="The name of the task to train selected in the list: ")
   parser.add_argument("--output_dir", default=None, type=str,
                     help="The output directory where the model predictions and checkpoints will be written.")

   parser.add_argument("--data_label_type", default='bal', type=str, help="bal or all")
   parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
   parser.add_argument("--train_data_type", default='bal', type=str, help="bal or all")
   parser.add_argument("--eval_data_type", default='bal', type=str, help="bal or all")
   parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")
   parser.add_argument("--use_layernorm", action='store_true', help="use_layernorm")
   parser.add_argument("--use_label_seq", action='store_true', help="use_label_seq")
   parser.add_argument("--use_pair", action='store_true', help="use_pair")
   parser.add_argument("--num_choice", default=2, type=int, help="num_choice")
   parser.add_argument("--max_seq_length", default=2, type=int, help="max lenght of text sequence")
   parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
   parser.add_argument("--img_feature_dim", default=2048, type=int)
   parser.add_argument("--code_voc", default=512, type=int)
   parser.add_argument("--img_feature_type", default="faster_r-cnn", type=str)
   #parser.add_argument("--num_choice", default=2, type=int)
   parser.add_argument("--classifier", default="mlp", type=str)
   parser.add_argument("--cls_hidden_scale", default=2, type=int)
   args = parser.parse_args()


   features = load_obj_tsv("/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/Oscar/github/Places/valid_obj36.tsv")
   config = BertConfig.from_pretrained("bert-base-uncased", num_labels=NUM_CLASSES, finetuning_task=args.task_name)

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

   dataset = Places365(args,"train",features,tokenizer)
   data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=12,
         pin_memory=True
   )

   batch = next(iter(data_loader))
    
 
   input_ids, attention_mask, segment_ids, labels, img_feats= batch     

   model = OscarForImageClassification(config)

  

   criterion = nn.CrossEntropyLoss()
   optim = AdamW(model.parameters(), lr=LR)



   print("***** Running training *****")
   print("  Num epochs: ", NUM_EPOCHS)
   print("  Batch size: ", BATCH_SIZE)

   for i in tqdm(range(NUM_EPOCHS), desc="epochs"):

      output = model(input_ids = input_ids,attention_mask=  attention_mask, position_ids = segment_ids, img_feats= img_feats)
      loss = criterion(output[0],labels)
      loss.backward()
      optim.step()
      print(loss)