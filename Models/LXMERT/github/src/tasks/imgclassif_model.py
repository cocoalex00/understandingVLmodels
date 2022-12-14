# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from .param import args as arguments
from src.lxrt.entry import LXRTEncoder
from src.lxrt.modeling import BertLayerNorm, GeLU
from pytorch_transformers.modeling_bert import BertPooler

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 2


class imgClassifModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            arguments,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_classes)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))

        logit = self.logit_fc(x)

        return logit