'''Some helper functions for PyTorch.'''
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

def get_mean_and_std(dataset, max_load=10000):
    """Compute the mean and std value of dataset."""
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im,_,_ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:,j,:,:].mean()
            std[j] += im[:,j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std

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


def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    box1 = change_box_order(box1, "xywh2xyxy")

    lt = tf.reduce_max([box1[:, :2], box2[:, :2]])  # [N,M,2]
    rb = tf.reduce_max([box1[:, 2:], box2[:, 2:]])  # [N,M,2]
    print(lt, rb)

    wh = tf.clip_by_value(rb-lt+1, 0, float('nan'))
    print(wh)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]+1) * (box1[:, 3]-box1[:, 1]+1)  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]+1) * (box2[:, 3]-box2[:, 1]+1)  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def draw_bboxes(image, boxes, labels):
    boxes = np.array(boxes, dtype=np.int32)
    for box, label in zip(boxes, labels):
        ymin, xmin, ymax, xmax = box
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
        #image = cv2.putText(image, str(label), (box[0]+15, box[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    return image

def draw_boxes(img, bboxes, classes, scores):
    if len(bboxes) == 0:
        return img

    #height, width, _ = img.shape
    width, height = img.size
    #image = Image.fromarray(img)
    image = img
    font = ImageFont.truetype(
        font='/root/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.4).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)

    for box, category, score in zip(bboxes, classes, scores):
        y1, x1, y2, x2 = [int(i) for i in box]

        p1 = (x1, y1)
        p2 = (x2, y2)

        label = '{} {:.1f}%   '.format(category, score * 100)
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - label_size[1]])

        color = np.array((0,255,0))
        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

        draw.rectangle(
            [tuple(text_origin),
             tuple(text_origin + label_size)],
            fill=tuple(color))

        draw.text(
            tuple(text_origin),
            label, fill=(0, 0, 0),
            font=font)

    del draw
    return np.array(image)

def draw_text_boxes(img, bboxes, color=(0,255,0)):
    if len(bboxes) == 0:
        return img

    width, height = img.size
    image = img

    draw = ImageDraw.Draw(image)
    thickness = (image.size[0] + image.size[1]) // 400
    
    for box in bboxes:
        y1, x1, y2, x2 = [int(i) for i in box]

        p1 = (x1, y1)
        p2 = (x2, y2)

        color = np.array(color)
        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

    del draw
    return image

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
  boxlist1 = change_box_order(boxlist1, "yxhw2yxyx")

  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

def bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        #jaccard = tfe_math.safe_divide(inter_vol, union_vol, 'jaccard')
        #return jaccard
        return tf.where(
            tf.greater(bboxes_vol, 0),
            tf.divide(inter_vol, bboxes_vol),
            tf.zeros_like(inter_vol),
            name='jaccard')
'''
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
'''
