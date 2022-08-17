import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert, VisualLinguisticBertMVRCHeadTransform
from torch.nn import GELU
# from pytorch_transformers.modeling_bert import BertPooler

BERT_WEIGHTS_NAME = 'pytorch_model.bin'

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=False)

            self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        
        self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)


        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        self.language_pretrained_model_path = language_pretrained_model_path
        
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path)

        # self.hm_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
        # self.hi_out = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)

        transform = VisualLinguisticBertMVRCHeadTransform(config.NETWORK.VLBERT)
        # linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.DATASET.ANSWER_VOCAB_SIZE)
        # self.final_mlp = nn.Sequential(
        #     transform,
        #     nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
        #     linear
        # )

        # self.final_mlp = torch.nn.Sequential(
        #         torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
        #         torch.nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
        #         torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, config.DATASET.ANSWER_VOCAB_SIZE),
        # )

        self.final_mlp = nn.Sequential(
            transform,
            nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size * 2),
            #nn.LayerNorm(config.NETWORK.VLBERT.hidden_size * 2, eps=1e-6),
            torch.nn.ReLU(inplace=True),
            nn.Linear(config.NETWORK.VLBERT.hidden_size* 2, config.NETWORK.VLBERT.hidden_size *3 ),
            torch.nn.ReLU(inplace=True),
            nn.Linear(config.NETWORK.VLBERT.hidden_size * 3, config.DATASET.ANSWER_VOCAB_SIZE),
        )

        print(self.final_mlp)
        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        # self.hm_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hm_out.bias.data.zero_()
        # self.hi_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hi_out.bias.data.zero_()
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        if self.config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            language_pretrained = torch.load(self.language_pretrained_model_path)
            mlm_transform_state_dict = {}
            pretrain_keys = []
            for k, v in language_pretrained.items():
                if k.startswith('cls.predictions.transform.'):
                    pretrain_keys.append(k)
                    k_ = k[len('cls.predictions.transform.'):]
                    if 'gamma' in k_:
                        k_ = k_.replace('gamma', 'weight')
                    if 'beta' in k_:
                        k_ = k_.replace('beta', 'bias')
                    mlm_transform_state_dict[k_] = v
            print("loading pretrained classifier transform keys: {}.".format(pretrain_keys))
            self.final_mlp[0].load_state_dict(mlm_transform_state_dict)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        #turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def train_forward(self,
                      image,
                      boxes,
                      im_info,
                      text,
                      ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :,0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        #print(box_mask.shape)
        # print(boxes.shape)


        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        ############################################

        # print(obj_reps['obj_reps'].shape)

        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        # print(text_visual_embeddings.shape)

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2

        ###########################################

        # Visual Linguistic BERT

        hidden_states_t, hidden_states_v, pooled_output = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      output_all_encoded_layers=False,
                                      output_text_and_object_separately=True)
        #_batch_inds = torch.arange(question.shape[0], device=question.device)

        # we only want the hidden states corresponding to the whole image  features
        #   input of the following form:
        #       [CLS] + [SEP] + [Whole image feature] + [ROI features]*36 + [END]
        hm = hidden_states_t[:,-1, :]

        #print(hidden_states_v.shape)
        
        capaz = torch.sum(hidden_states_v,1) / 37
        #print(hm[0,:] == hm[1,:])

        #print(hm.shape)

        # print(hidden_states_t.shape)
        #print(hidden_states_v.shape)
        # print(hm.shape)
        # hm = F.tanh(self.hm_out(hidden_states[_batch_inds, ans_pos]))
        # hi = F.tanh(self.hi_out(hidden_states[_batch_inds, ans_pos + 2]))

        ###########################################
        #outputs = {}

        # classifier
        # logits = self.final_mlp(hc * hm * hi)
        # logits = self.final_mlp(hc)
        logits = self.final_mlp(hm)

        #print(logits.shape)

        #print(logits.shape)

        # loss
        #ans_loss = F.binary_cross_entropy_with_logits(logits, label) * label.size(1)

        #outputs.update({'label_logits': logits,
        #                'label': label,
        #                'ans_loss': ans_loss})

        #loss = ans_loss.mean()

        return  logits#outputs, loss

    def inference_forward(self,
                          image,
                          boxes,
                          im_info,
                          question):

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None)

        question_ids = question
        question_tags = question.new_zeros(question_ids.shape)
        question_mask = (question > 0.5)

        answer_ids = question_ids.new_zeros((question_ids.shape[0], 1)).fill_(
            self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
        answer_mask = question_mask.new_zeros(answer_ids.shape).fill_(1)
        answer_tags = question_tags.new_zeros(answer_ids.shape)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ans_pos = self.prepare_text_from_qa(question_ids,
                                                                                                       question_tags,
                                                                                                       question_mask,
                                                                                                       answer_ids,
                                                                                                       answer_tags,
                                                                                                       answer_mask)
        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, hc = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      output_all_encoded_layers=False)

        hm = hidden_states[2:, :]
        # hm = F.tanh(self.hm_out(hidden_states[_batch_inds, ans_pos]))
        # hi = F.tanh(self.hi_out(hidden_states[_batch_inds, ans_pos + 2]))

        ###########################################
        outputs = {}

        # classifier
        # logits = self.final_mlp(hc * hm * hi)
        # logits = self.final_mlp(hc)
        logits = self.final_mlp(hm)

        outputs.update({'label_logits': logits})

        return outputs
