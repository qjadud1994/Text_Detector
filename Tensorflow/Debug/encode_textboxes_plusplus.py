import numpy as np
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]=""

anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
aspect_ratios = [1, 2, 3, 5, 1/2, 1/3, 1/5]
num_anchors = len(aspect_ratios)* 2

input_shape = np.array([608, 608])


def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
    Args:
      boxes: (tensor) bounding boxes, sized [num anchors, 4].
    Returns:
      (tensor) converted bounding boxes, sized [num anchor, 4].
    '''

    if order is 'yxyx2yxhw':
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        new_boxes = tf.concat([y,x,h,w], axis=1)

    elif order is 'yxhw2yxyx':
        y, x, h, w = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        x_min = x - w/2
        x_max = x + w/2
        y_min = y - h/2
        y_max = y + h/2
        new_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=1)

    elif order is 'xyxy2yxyx':
        x_min, y_min, x_max, y_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        new_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=1)

    elif order is 'yxyx2xyxy':
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        new_boxes = tf.concat([x_min, y_min, x_max, y_max], axis=1)
        
    elif order is "yxhw2quad":
        """rect : [num_boxes, 4] #yxhw, / quad : [num_boxes, 8]"""
        y0, x0, h0, w0 = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        new_boxes = tf.concat([y0-h0/2, x0-w0/2, 
                               y0-h0/2, x0+w0/2, 
                               y0+h0/2, x0+w0/2, 
                               y0+h0/2, x0-w0/2], axis=1)

    elif order is "quad2yxyx":
        """quad : [num_boxes, 8] / rect : [num_boxes, 4] #yxyx"""
        boxes = tf.reshape(boxes, (-1, 4, 2))
        new_boxes = tf.concat([tf.reduce_min(boxes[:, :, 0:1], axis=1),
                               tf.reduce_min(boxes[:, :, 1:2], axis=1),
                               tf.reduce_max(boxes[:, :, 0:1], axis=1),
                               tf.reduce_max(boxes[:, :, 1:2], axis=1)], axis=1)
        
    return new_boxes

def area(boxlist, scope=None):
  """Computes area of boxes.
  Args:
    boxlist: BoxList holding N boxes following order [ymin, xmin, ymax, xmax]
    scope: name scope.
  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """

  with tf.name_scope(scope, 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1, num_or_size_splits=4, axis=1)

    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2, num_or_size_splits=4, axis=1)

    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.
  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.
  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  #boxlist1 = change_box_order(boxlist1, "yxhw2yxyx")

  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

    

gt_quad_boxes = tf.random_normal([3, 8]) + 3  #yxyxyxyx
labels = tf.zeros([3]) + 3

anchor_rect_boxes = tf.random_normal([11, 4]) + 3 #ywhw


anchor_boxes_hw = tf.tile(anchor_rect_boxes[:, 2:4], [1, 4])

gt_rect_boxes = change_box_order(gt_quad_boxes, "quad2yxyx")

anchor_quad_boxes = change_box_order(anchor_rect_boxes, "yxhw2quad")

'''boxes : yxyx , anchor_boxes : yxhw'''
print("anchor_rect_boxes : ", anchor_rect_boxes.shape)
ious = iou(anchor_rect_boxes, gt_rect_boxes)

max_ids = tf.argmax(ious, axis=1, name="encode_argmax")
max_ious = tf.reduce_max(ious, axis=1)

#Each anchor box matches the largest iou with the gt box
gt_quad_boxes = tf.gather(gt_quad_boxes, max_ids)   
gt_rect_boxes = tf.gather(gt_rect_boxes, max_ids)

# rect-anchor box offset
gt_rect_boxes = change_box_order(gt_rect_boxes, "yxyx2yxhw")
loc_rect_yx = (gt_rect_boxes[:, :2] - anchor_rect_boxes[:, :2]) / anchor_rect_boxes[:, 2:]
loc_rect_hw = tf.log(gt_rect_boxes[:, 2:] / anchor_rect_boxes[:, 2:])

loc_quad_yx = (gt_quad_boxes - anchor_quad_boxes) / anchor_boxes_hw

loc_targets = tf.concat([loc_rect_yx, loc_rect_hw, loc_quad_yx], 1)
cls_targets = 1 + tf.gather(labels, max_ids)    # labels : (0~19) + 1 -> (1~20)
#cls_targets = tf.gather(labels, max_ids)    # VOC labels 1~20

# iou < 0.4 : background(0)  /  0.4 < iou < 0.5 : ignore(-1)
cls_targets = tf.where(tf.less(max_ious, 0.5), -tf.ones_like(cls_targets), cls_targets)
cls_targets = tf.where(tf.less(max_ious, 0.4), tf.zeros_like(cls_targets), cls_targets)

def Print(tensor, name, sess):
    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([anchor_rect_boxes, anchor_quad_boxes, ious, max_ids, gt_quad_boxes, gt_rect_boxes, anchor_boxes_hw, loc_quad_yx, loc_targets, cls_targets],
          ["anchor_rect_boxes", "anchor_quad_boxes", "ious", "max_ids", "gt_quad_boxes", "gt_rect_boxes", "anchor_boxes_hw", "loc_quad_yx", "loc_targets", "cls_targets"],
          sess)


