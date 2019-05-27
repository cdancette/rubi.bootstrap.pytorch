import torch
import torch.nn as nn
from block.models.networks.mlp import MLP
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 


class RUBi(nn.Module):
    """
    Same parameters as SAN
    """
    def __init__(self, model, output_size, classif, end_classif=True):
        super().__init__()
        self.model = model
        self.c_1 = MLP(**classif)
        self.end_classif = end_classif
        if self.end_classif:
            self.c_2 = nn.Linear(output_size, output_size)


    def forward(self, batch):
        out = {}
        # model prediction
        net_out = self.model(batch)
        logits = net_out['logits']

        q_embedding = net_out['q_emb']  # N * q_emb
        q_embedding = grad_mul_const(q_embedding, 0.0)  #  don't packpropagate through question encoder
        q_pred = self.c_1(q_embedding)
        fusion_pred = logits * torch.sigmoid(q_pred)

        if self.end_classif:
            q_out = self.c_2(q_pred)
        else:
            q_out = q_pred

        out['logits'] = net_out['logits']
        out['logits_rubi'] = fusion_pred
        out['logits_q'] = q_out
        return out

    def process_answers(self, out, key=''):
        out = self.model.process_answers(out)
        out = self.model.process_answers(out, key='_rubi')
        out = self.model.process_answers(out, key='_q')
        return out

