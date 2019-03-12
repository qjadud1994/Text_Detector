import torch
import torch.nn as nn

from fpn import FPN50
from torch.autograd import Variable


class RetinaNet(nn.Module):

    def __init__(self, num_classes=1):
        super(RetinaNet, self).__init__()
        self.num_anchors = 7*2  # vertical offset -> *2
        self.num_classes = num_classes
        self.fpn = FPN50()
        self.loc_head = self._make_head(self.num_anchors*8)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms: #for all FPN features
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)                 # [N,H*W*num_anchors, 8]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,H*W*num_anchors, num_classes]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=(3, 5), stride=1, padding=(1, 2)))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
