from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import block
from .pairwise import Pairwise

class MuRelCell(nn.Module):

    def __init__(self,
            residual=False,
            fusion={},
            pairwise={},
            pairwise_type="regular",
            keep_topk=0,
            topk_criterion="l2",
            return_all=False):
        """
        keep_topk: will keep only the top k regions in every image. 
            The choice is based on L2 norm of the fused vector (image + question).
            This should discard regions that don't match the question. 
        topk_criterion: in [l2, argmax]
        pairwise_type: test parameter to compare various pairwise implementations.
            in [regular, 1_loop, 2_loops]
        return_all: return mm before and after residual
        """
        super(MuRelCell, self).__init__()
        self.topk_criterion = topk_criterion
        self.residual = residual
        self.fusion = fusion
        self.pairwise = pairwise
        self.keep_topk = keep_topk
        self.fusion_module = block.factory_fusion(self.fusion)
        if self.pairwise:
            self.pairwise_module = Pairwise(**pairwise)
        self.buffer = None
        self.return_all = return_all

    def forward(self, q_expand, mm, coords=None, nb_regions=None):
        mm_new = self.process_fusion(q_expand, mm)

        if self.pairwise:
            if self.keep_topk:
                # keep only topk regions based on L2 norm
                if self.topk_criterion == "l2":
                    criterion = torch.norm(mm_new, dim=2) # size (batch, objects)
                elif self.topk_criterion == "argmax":
                    criterion, _ = torch.max(mm_new, dim=2)
                max, indices = torch.topk(criterion, self.keep_topk, dim=1) # size: batch, k
                mm_new_topk = mm_new[torch.arange(mm.size(0)), indices.t()].transpose(1, 0)
                coords = coords[torch.arange(mm.size(0)), indices.t()].transpose(1, 0)
                nb_regions = [self.keep_topk] * mm.shape[0]
                mm_new_topk = self.pairwise_module(mm_new_topk, coords, nb_regions)
                mm_new[torch.arange(mm.size(0)), indices.t()] = mm_new_topk.transpose(1, 0)
            else:   
                mm_new = self.pairwise_module(mm_new, coords, nb_regions)

        if self.residual:
            mm_new = mm_new + mm
            out['mm_res'] = mm_new
        
        if self.buffer is not None:
            self.buffer['mm'] = mm.data.cpu()
            self.buffer['mm_new'] = mm_new.data.cpu()

        return mm_new

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm
