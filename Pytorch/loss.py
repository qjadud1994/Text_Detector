from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding, one_hot_v3
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.num_classes = 1

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), self.num_classes+1)  # [N,21]

        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        t = one_hot_embedding(y.data.cpu(), self.num_classes+1)
        t = t[:,1:]
        t = Variable(t).cuda()
        
        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.float().sum()
        
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,8]
        masked_loc_preds = loc_preds[mask].view(-1,8)      # [#pos,8]
        masked_loc_targets = loc_targets[mask].view(-1,8)  # [#pos,8]
        
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        loc_loss *= 0.5  # TextBoxes++ has 8-loc offset
        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        return loc_loss/num_pos, cls_loss/num_pos

    
    
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class OHEM_loss(nn.Module):
    def __init__(self):
        super(OHEM_loss, self).__init__()
        self.num_classes = 2
        self.negpos_ratio = 3

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        cls_targets = cls_targets.clamp(0, 1)   #remove ignore (-1)
        pos = cls_targets > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        masked_loc_preds = loc_preds[pos_idx].view(-1, 8)
        masked_loc_targets = loc_targets[pos_idx].view(-1, 8)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        # Compute max conf across batch for hard negative mining
        num = loc_preds.size(0)
        batch_conf = cls_preds.view(-1, self.num_classes)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, cls_targets.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(cls_preds)
        neg_idx = neg.unsqueeze(2).expand_as(cls_preds)
        
        conf_p = cls_preds[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = cls_targets[(pos+neg).gt(0)]
        cls_loss = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.float().sum()
        loc_loss /= N
        cls_loss /= N
        return loc_loss, cls_loss

    
def Debug():
    loc_preds = torch.randn((2, 2, 8))
    loc_targets = torch.randn((2, 2, 8))
    
    cls_preds = torch.randn((2, 2, 2))
    cls_targets = torch.randint(0, 2, (2, 2)).type(torch.LongTensor)
    
    print(cls_targets.data)
    
    ohem = OHEM_loss()
    ohem.forward(loc_preds, loc_targets, cls_preds, cls_targets)

#Debug()