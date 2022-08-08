from os import O_EXCL
from re import I
import torch

import torch.nn as nn 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

MAX_VQA_LENGTH = 2
NUM_CLASSES = 365
LR= 0.000002
NUM_EPOCHS= 300
BATCH_SIZE = 2
WEIGHT_PATH = "/mnt/c/Users/aleja/Desktop/MSc Project/Implementation/Models/LXMERT/github/snap/pretrained/model"
DATAPATH = "/mnt/c/Users/aleja/Desktop/MSc Project/images"
from transformers import  DataCollatorForLanguageModeling

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datasets.Places_dataset import Places365


from vilt.datasets.sapeplaces import PlacesDatasetBase

from vilt.datamodules.multitask_datamodule import MTDataModule

from transformers import DataCollatorForLanguageModeling, BertTokenizer, AdamW
import functools

from torch.nn import CrossEntropyLoss


@ex.automain
def main(_config):
   #_config = copy.deepcopy(_config)
   #pl.seed_everything(_config["seed"])
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   collator = DataCollatorForLanguageModeling(tokenizer,False)

   dm = Places365(DATAPATH, split="val")

   collate = functools.partial(
            dm.collate, mlm_collator=collator,
        )
   dataloader = DataLoader(dm,
                           batch_size=2,
                           collate_fn=collate
                           )

   batch = next(iter(dataloader))

   model = ViLTransformerSS(_config,"places")

   criterion = nn.CrossEntropyLoss()
   optim = AdamW(model.parameters(), lr=LR)

   

   print(batch["label"])


      
   print("***** Running training *****")
   print("  Num epochs: ", NUM_EPOCHS)
   print("  Batch size: ", BATCH_SIZE)

   for i in tqdm(range(NUM_EPOCHS), desc="epochs"):

      #optim.zero_grad()

      output = model(batch)
      label = torch.tensor([l.item() for l in batch["label"]])

      print(label)
      loss = criterion(output["imgcls_logits"],label)
      loss.backward()
      optim.step()
      print(loss)


