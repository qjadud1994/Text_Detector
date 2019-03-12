'''Encode object boxes and labels.'''
import math
import torch
import numpy as np

from utils import meshgrid, box_iou, change_box_order, softmax
from nms_poly import non_max_suppression_poly

class DataEncoder:
    def __init__(self, cls_thresh=0.3, nms_thresh=0.1):
        self.anchor_areas = [16*16., 32*32., 64*64., 128*128., 256*256, 512*512.]  # v3
        self.aspect_ratios = [1., 2., 3., 5., 1./2., 1./3., 1./5.]                  # v3

        #self.anchor_areas = [30*30., 70*70., 120*120., 250*250., 320*320. ,450*450.]  #v5
        #self.aspect_ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 0.5, 0.2]                      #v5

        self.anchor_wh = self._get_anchor_wh()

        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                anchor_h = math.sqrt(s/ar)
                anchor_w = ar * anchor_h
                anchor_wh.append([anchor_w, anchor_h])

        num_fms = len(self.anchor_areas)
        return torch.FloatTensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+2)).ceil() for i in range(num_fms)]  # p2 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            fm_w *= 2  # add vertical offset
            xy = meshgrid(fm_w,fm_h) + 0.5 
            xy = (xy*grid_size).view(fm_w,fm_h,1,2).expand(fm_w,fm_h,len(self.aspect_ratios),2)

            wh = self.anchor_wh[i].view(1,1,len(self.aspect_ratios),2).expand(fm_w,fm_h,len(self.aspect_ratios),2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, gt_quad_boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        TextBoxes++ quad_box encoder:
          tx_n = (x_n - anchor_x) / anchor_w
          ty_n = (y_n - anchor_y) / anchor_h

        Args:
          gt_quad_boxes: (tensor) bounding boxes of (xyxyxyxy), sized [#obj, 8].
          labels: (tensor) object class labels, sized [#obj, ].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,8].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)

        anchor_rect_boxes = self._get_anchor_boxes(input_size)                 #(num_anchor, 8)
        anchor_quad_boxes = change_box_order(anchor_rect_boxes, "xywh2quad")   #(num_anchor, 4)

        gt_rect_boxes = change_box_order(gt_quad_boxes, "quad2xyxy")

        ious = box_iou(anchor_rect_boxes, gt_rect_boxes)
        max_ious, max_ids = ious.max(1)

        #Each anchor box matches the largest iou with the gt box
        gt_quad_boxes = gt_quad_boxes[max_ids]  #(num_gt_boxes, 8)
        gt_rect_boxes = gt_rect_boxes[max_ids]  #(num_gt_boxes, 4)

        # for Rectangle boxes -> using in TextBoxes
        #gt_rect_boxes = change_box_order(gt_rect_boxes, "xyxy2xywh")
        #loc_rect_yx = (gt_rect_boxes[:, :2] - anchor_rect_boxes[:, :2]) / anchor_rect_boxes[:, 2:]
        #loc_rect_hw = torch.log(gt_rect_boxes[:, 2:] / anchor_rect_boxes[:, 2:])

        # for Quad boxes -> using in TextBoxes++
        anchor_boxes_hw = anchor_rect_boxes[:, 2:4].repeat(1, 4)
        loc_quad_yx = (gt_quad_boxes - anchor_quad_boxes) / anchor_boxes_hw

        #loc_targets = torch.cat([loc_rect_yx, loc_rect_hw, loc_quad_yx], dim=1) # (num_anchor, 12)
        loc_targets = loc_quad_yx
        cls_targets = labels[max_ids]

        cls_targets[max_ious<0.5] = -1    # ignore (0.4~0.5) : -1
        cls_targets[max_ious<0.4] = 0     # background (0.0~0.4): 0
                                          # positive (0.5~1.0) : 1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 8].
          cls_preds: (tensor) predicted class labels, sized [#anchors, ].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,8].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        
        anchor_rect_boxes = self._get_anchor_boxes(input_size).cuda()
        anchor_quad_boxes = change_box_order(anchor_rect_boxes, "xywh2quad")

        quad_boxes = anchor_quad_boxes + anchor_rect_boxes[:, 2:4].repeat(1, 4) * loc_preds  # [#anchor, 8]
        quad_boxes = torch.clamp(quad_boxes, 0, input_size[0])
        
        score, labels = cls_preds.sigmoid().max(1)          # focal loss
        #score, labels = softmax(cls_preds).max(1)          # OHEM+softmax
        
        # Classification score Threshold
        ids = score > self.cls_thresh
        ids = ids.nonzero().squeeze()   # [#obj,]
        
        score = score[ids]
        labels = labels[ids]
        quad_boxes = quad_boxes[ids].view(-1, 4, 2)
        
        quad_boxes = quad_boxes.cpu().data.numpy()
        score = score.cpu().data.numpy()
        
        if len(score.shape) is 0:
            return quad_boxes, labels, score
        else:
            keep = non_max_suppression_poly(quad_boxes, score, self.nms_thresh)
            return quad_boxes[keep], labels[keep], score[keep]

def debug(): 
    encoder = DataEncoder()

    anchor_wh = encoder._get_anchor_wh()

    input_size = 32
    input_size =  torch.Tensor([input_size,input_size])
    anchor = encoder._get_anchor_boxes(input_size)

    print("anchor.size() : ", anchor.size())
    
    for i in anchor:
        print(i)
    
    exit()

    test = torch.randn((3, 8))
    #test2 = torch.reshape(test, (-1, 4, 2))
    test2 = test.view((-1, 4, 2))
    print("test : ", test, test.size())
    print("test2 : ", test2, test2.size())


    gt_quad_boxes = torch.randn((1400, 8))
    labels = torch.randn((1400, 1))
    result_encode = encoder.encode(gt_quad_boxes, labels, input_size)
    print(result_encode[0].size())
    print(result_encode[1].size())
    

#debug()
