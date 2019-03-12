import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

loc_preds = tf.random_normal([5, 4])
cls_preds = tf.random_normal([5, 10])
anchor_boxes = tf.random_normal([5, 4])

loc_preds2 = tf.random_normal([1, 5, 4])
cls_preds2 = tf.random_normal([2, 5, 10])

print(len(loc_preds.shape.as_list()))
print(len(loc_preds2.shape.as_list()))

cls_thresh = 0.8
max_detect = 300
nms_thresh = 0.5

loc_yx = loc_preds[:, :2]
loc_hw = loc_preds[:, 2:]

yx = loc_yx * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
hw = tf.exp(loc_hw) * anchor_boxes[:, 2:]

boxes = tf.concat([yx-hw/2, yx+hw/2], axis=1)  # [#anchors,4]

cls_preds = tf.nn.sigmoid(cls_preds)
labels = tf.argmax(cls_preds, axis=1, name="decode_argmax")
score = tf.reduce_max(cls_preds, axis=1)

ids = tf.where(score > cls_thresh)
ids = tf.squeeze(ids, axis=1)

boxes2 = tf.gather(boxes, ids)
score2 = tf.gather(score, ids)

keep = tf.image.non_max_suppression(boxes2, score2, max_detect, nms_thresh)

boxes3 = tf.gather(boxes2, keep, name="111111")
labels3 = tf.gather(labels, keep, name="222222")

def Print(tensor, name, sess):
    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([tf.shape(loc_preds), tf.shape(loc_preds2), tf.size(loc_preds), tf.size(loc_preds2)], 
          ["tf.shape(loc_preds)", "tf.shape(loc_preds2)", "tf.size(loc_preds)", "tf.size(loc_preds2)"], sess)

    Print([loc_preds, cls_preds, boxes, labels, 
           score, ids, boxes2, score2, keep, boxes3, labels3],
          ["loc_preds", "cls_preds", "boxes", "labels", 
           "score", "ids", "boxes2", "score2", "keep", "boxes3", "labels3"],
          sess)


