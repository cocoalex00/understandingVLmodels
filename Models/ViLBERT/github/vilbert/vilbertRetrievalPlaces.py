# Copyright (c) Alejandro Hernandez Diaz.

# This source code implements the ViLBERT model class in order to be fine tuned to Image classification.
import torch
import torch.nn as nn

from .vilbert import VILBertForVLTasks, BertConfig, BertLayerNorm , GeLU




## VILBert for Image Classification, adapted from its original implementation
##  - Link: https://github.com/facebookresearch/vilbert-multi-task

class VILBertForRetrieval(torch.nn.Module):

    def __init__(self,config_file,from_pretrained,device, default_gpu=True):
        """ Vilbert Model adapted to perform image classification.

            config_file: path to the json file containing BERT's configuration
            num_labels: number of classes in the dataset to use
            from_pretrained: path to the .bin file containing the pre-trained weights
            default_gpu: 
        """
        super(VILBertForRetrieval,self).__init__()
        # Bert configuration
        self.config = BertConfig.from_json_file(config_file)
        #ViLBERT model
        self.vilbertBase = VILBertForVLTasks.from_pretrained(
            from_pretrained,
            config=self.config,
            num_labels=1,
            device=device,
        )

        # Last fully connected layer (design taken from original paper)
        self.ImageClassifFC = nn.Sequential(
            nn.Linear(self.config.bi_hidden_size, self.config.bi_hidden_size * 2),
            GeLU(),
            BertLayerNorm(self.config.bi_hidden_size * 2 , eps=1e-12),
            nn.Linear(self.config.bi_hidden_size * 2 , 1),
        )


    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        co_attention_mask=None,
        task_ids=None,
        output_all_encoded_layers=False,
        output_all_attention_masks=False,
    ):

        # Forward pass through the baseline ViLBERT
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, all_attention_mask, sequence_output_v, pooled_output_v, pooled_output = self.vilbertBase(
            input_txt, input_imgs, image_loc, token_type_ids, attention_mask, image_attention_mask, co_attention_mask, task_ids, output_all_encoded_layers, output_all_attention_masks,
        )

        # alignmnent logits (use overall pooled output which is an overall represerntation of the input)
        #print(pooled_output)
        alignment_logits= self.ImageClassifFC(pooled_output)
        
       # print(pooled_output_v.shape)

        return (
            alignment_logits,
            vil_prediction,
            vil_prediction_gqa,
            vil_logit,
            vil_binary_prediction,
            vil_tri_prediction,
            vision_prediction,
            vision_logit,
            linguisic_prediction,
            linguisic_logit,
            all_attention_mask,
            sequence_output_v, ## added by Alejandro Hernandez Díaz
            pooled_output_v ##
        )