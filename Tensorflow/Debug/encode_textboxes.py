import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""


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



boxes = tf.random_normal([3, 4])
labels = tf.random_normal([3])

anchor_boxes = tf.random_normal([11, 4])

ious = iou(anchor_boxes, boxes)

max_ids = tf.argmax(ious, axis=1, name="encode_argmax")
max_ious = tf.reduce_max(ious, axis=1)

boxes = tf.gather(boxes, max_ids)
labels = tf.gather(labels, max_ids)

cls_targets = tf.where(tf.not_equal(max_ious, 0.0), -tf.ones_like(labels), labels)

def Print(tensor, name, sess):
    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([ious, max_ids, max_ious, boxes, labels, cls_targets],
          ["ious", "max_dis", "max_ious", "boxes", "labels", "cls_targets"],
          sess)


