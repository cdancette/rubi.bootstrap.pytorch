from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP

from .murel_cell import MuRelCell
from .utils import mask_softmax

class BaselineNet(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            cell={},
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.cell = cell
        self.cell_params = cell
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.cell = MuRelCell(**cell)

        if self.classif['mlp']['dimensions'][-1] != len(self.aid_to_ans):
            Logger()(f"Warning, the classif_mm output dimension ({self.classif['mlp']['dimensions'][-1]})" 
             f"doesn't match the number of answers ({len(self.aid_to_ans)}). Modifying the output dimension.")
            self.classif['mlp']['dimensions'][-1] = len(self.aid_to_ans) 

        self.classif_module = MLP(**self.classif['mlp'])

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)

      
    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def set_pairs_ids(self, n_regions, bsize, device='cuda'):
        if self.shared and self.cell.pairwise:
            self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
        else:
            for i in self.n_step:
                if self.cells[i].pairwise:
                    self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)

    def forward(self, batch):
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]

        out = {}

        q = self.process_question(q, l,)
        out['q_emb'] = q
        q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize*n_regions, -1)

        mm = self.cell(q_expand, v, c, nb_regions)
        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)

        out['mm'] = mm
        out['mm_argmax'] = mm_argmax

        logits = self.classif_module(mm)

        out['logits'] =  logits

        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)


        q, _ = txt_enc.rnn(q_emb)

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = txt_enc._select_last(q, l)

        return q

    def process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out
