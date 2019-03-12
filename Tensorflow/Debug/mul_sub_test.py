import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes=20


tensor = tf.random_normal((2, 4, 2))

mul_1 = 2 * tensor
mul_2 = tf.scalar_mul(2, tensor)

sub_1 = tensor - 1
sub_2 = tensor - tf.ones_like(tensor)

div_1 = tensor / 10.0
div_2 = tf.truediv(tensor, 10.0)

sum_1 = tf.reduce_sum(tensor)
sum_2 = tf.reduce_sum(sum_1)
sum_3 = tf.reduce_sum(tensor, axis=1)
sum_4 = tf.reduce_sum(sum_3)

cls_gt = tf.constant( [[0, 19], [0, 20]] )
cls_pred = tf.constant( [[0, 19], [0, 19]] )

cls_gt_one_hot = tf.one_hot(cls_gt, 21)
filter_cls_gt_one_hot = cls_gt_one_hot[:, 1:]

cls_pred_one_hot = tf.one_hot(cls_pred, 20)

label_one_hot = tf.one_hot(indices=cls_gt-1, depth=20, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)

def Print(tensor, name, sess):

    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([tensor, mul_1, mul_2, sub_1, sub_2, tf.rank(tensor), div_1, div_2, sum_1, sum_2, sum_3, sum_4], 
          ["tensor", "mul_1", "mul_2", "sub_1", "sub_2", "size", "div_1", "div_2", "sum_1", "sum_2", "sum_3", "sum_4"],
          sess)

    Print([cls_gt, cls_pred, cls_gt_one_hot, filter_cls_gt_one_hot, cls_pred_one_hot, label_one_hot], 
          ["cls_gt", "cls_pred", "cls_gt_one_hot", "filter_cls_gt_one_hot", "cls_pred_one_hot", "label_one_hot"], sess)

