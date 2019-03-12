import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes=20

#loc_preds, cls_preds = tf.random_normal((8, 69354, 4)), tf.random_normal((8, 69354, 20))
#loc_gt, cls_gt = tf.random_normal((8, 69354, 4)), tf.random_normal((3,4)) #tf.random_normal((8, 69354))

loc_preds, cls_preds = tf.random_normal((2, 3, 4)), tf.random_normal((2, 3, 10))
loc_gt, cls_gt = tf.random_normal((2, 3, 4)), tf.random_normal((2, 3))

_cls_gt = tf.greater(cls_gt, 0, name="1")

valid_anchor_indices2 = tf.where(_cls_gt, name="3")
valid_anchor_indices = tf.squeeze(tf.where(_cls_gt), name="2")

gt_anchor_nums = tf.shape(valid_anchor_indices)[0]

"""Location Regression loss"""
# skip negative and ignored anchors
valid_loc_preds = tf.gather_nd(loc_preds, valid_anchor_indices, name="5")
valid_loc_gt = tf.gather_nd(loc_gt, valid_anchor_indices, name="7")

index = valid_anchor_indices
out = tf.gather_nd(loc_preds, index, name="8")

"""Classification loss"""
valid_cls_indices = tf.where(tf.greater(cls_gt, -1))

# skip ignored anchors (iou belong to 0.4 to 0.5)
valid_cls_preds = tf.gather_nd(cls_preds, valid_cls_indices)
valid_cls_gt = tf.gather_nd(cls_gt, valid_cls_indices)

valid_cls_gt = tf.cast(valid_cls_gt * 100, tf.int32)
#valid_cls_gt = tf.constant([0, 1, 9, 10])

gt_cls_ont_hot = tf.one_hot(valid_cls_gt, 11) #, dtype=tf.float32)
filter_gt_cls_one_hot = gt_cls_ont_hot[:, 1:]  # remove background

#### focal loss

x = tf.constant([[0,0,0,1], [0,0,1,0], [1,0,0,0], [0,0,0,0]], dtype=tf.float32) #pred
t = tf.constant([[0,0,0,1], [0,0,0,1], [1,0,0,0], [0,0,0,0]], dtype=tf.float32) #gt
xt = x * (2*t - 1)  # xt = x if t > 0 else -x
pt = tf.nn.sigmoid(2*xt + 1)

alpha = 0.25
w = alpha * t + (1 - alpha) * (1 - t)

loss = -w * tf.log(pt) / 2
loss_sum = tf.reduce_sum(loss)

def Print(tensor, name, sess):

    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([cls_gt, valid_anchor_indices, valid_anchor_indices2, gt_anchor_nums, loc_preds, 
           valid_loc_preds],
          ["cls_gt", "valid_anchor_indices", "valid_anchor_indices2", "gt_anchor_nums", 
           "loc_preds", "valid_loc_preds"], 
          sess)

    print("@" * 40)

    Print([cls_preds, cls_gt, valid_cls_indices, valid_cls_preds, valid_cls_gt, gt_cls_ont_hot, filter_gt_cls_one_hot], 
          ["cls_preds", "cls_gt", "valid_cls_indices", "valid_cls_preds", "valid_cls_gt", "gt_cls_one_hot", "filter_gt_cls_one_hot"], sess)


    print("@" * 40)
    Print([x, t, xt, pt,  w, loss, loss_sum], 
          ["x", "t", "xt", "pt",  "w", "loss", "loss_sum"], sess)
