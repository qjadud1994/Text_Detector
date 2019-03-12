import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

tensor = tf.random_normal((2, 2, 3))

mean1 = tf.constant((0.485, 0.456, 0.406), dtype=tf.float32)
var1 = tf.constant((0.229, 0.224, 0.225), dtype=tf.float32)
image1 = (tensor - mean1) / var1

mean2 = (0.485, 0.456, 0.406)
var2 = (0.229, 0.224, 0.225)
image2 = (tensor - mean2) / var2

mean3 = ((0.485, 0.456, 0.406))
var3 = ((0.229, 0.224, 0.225))
image3 = (tensor - mean3) / var3


def Print(tensor, name, sess):

    out = sess.run(tensor)

    for n, o in zip(name, out):
        print("\nname : ", n)
        print(o)
        print(o.shape)
        print("=" * 40)

with tf.Session() as sess:
    Print([image1, image2, image3], 
          ["image1", "image2", "image3"],
          sess)

