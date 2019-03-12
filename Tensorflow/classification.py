import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import cv2, os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow.contrib import learn
from Detector.RetinaNet import RetinaNet
from utils.bbox import draw_boxes

mode = learn.ModeKeys.INFER

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.WARN)
tf.app.flags.DEFINE_string('f', '', 'kernel')
#### Input pipeline
tf.app.flags.DEFINE_integer('input_size', 224,
                            """Input size""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Train batch size""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Learninig rate""")
tf.app.flags.DEFINE_integer('num_input_threads', 4,
                            """Number of readers for input data""")
tf.app.flags.DEFINE_integer('num_classes', 1000,
                            """number of classes""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """The number of gpu""")
tf.app.flags.DEFINE_string('tune_from', 'results_test1/model.ckpt-70000',
                           """Path to pre-trained model checkpoint""")
#tf.app.flags.DEFINE_string('tune_from', 'results_test1/',
#                           """Path to pre-trained model checkpoint""")


#### Train dataset
tf.app.flags.DEFINE_string('train_path', '../data/mjsynth/train',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern', '*.tfrecord',
                           """File pattern for input data""")
### Validation dataset (during training)
tf.app.flags.DEFINE_string('valid_dataset','VOC',
                          """Validation dataset name""")
tf.app.flags.DEFINE_integer('valid_device', 0,
                           """Device for validation""")
tf.app.flags.DEFINE_integer('valid_batch_size', 8,
                            """Validation batch size""")
tf.app.flags.DEFINE_boolean('use_validation', True,
                            """Whether use validation or not""")
tf.app.flags.DEFINE_integer('valid_steps', 1000,
                            """Validation steps""")

#### Output Path
tf.app.flags.DEFINE_string('output', 'results_test2/model.ckpt-66000',
                           """Directory for event logs and checkpoints""")
#### Training config
tf.app.flags.DEFINE_float('cls_thresh', 0.5,
                            """thresh for class""")
tf.app.flags.DEFINE_float('nms_thresh', 0.5,
                            """thresh for nms""")
tf.app.flags.DEFINE_integer('max_detect', 300,
                            """num of max detect (using in nms)""")
tf.app.flags.DEFINE_string('tune_scope', '',
                           """Variable scope for training""")
tf.app.flags.DEFINE_integer('max_num_steps', 2**21,
                            """Number of optimization steps to run""")
tf.app.flags.DEFINE_boolean('verbose', True,
                            """Print log in tensorboard""")
tf.app.flags.DEFINE_boolean('use_profile', False,
                            """Whether use Tensorflow Profiling""")
tf.app.flags.DEFINE_boolean('use_debug', False,
                            """Whether use TFDBG or not""")
tf.app.flags.DEFINE_integer('save_steps', 2000,
                            """Save steps""")
tf.app.flags.DEFINE_integer('summary_steps', 50,
                            """Save steps""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """Moving Average dacay factor""")

img_dir = "/root/DB/VOC/VOC2012/JPEGImages/"
VOC = {1 : "motorbike", 2 : "car", 3 : "person", 4 : "bus", 5 : "bird", 6 : "horse", 7 : "bicycle", 8 : "chair", 9 : "aeroplane", 10 : "diningtable", 11 : "pottedplant", 12 : "cat", 13 : "dog", 14 : "boat", 15 : "sheep", 16 : "sofa", 17 : "cow", 18 : "bottle", 19 : "tvmonitor", 20 : "train"}


def get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    return config

def load_pytorch_weight():
    from torch import load

    pt_load = load("weights/resnet50.pth")
    reordered_weights = {}
    pre_train_ops = []

    for key, value in pt_load.items():
        try:
            reordered_weights[key] = value.data.numpy()
        except:
            reordered_weights[key] = value.numpy()

    weight_names = list(reordered_weights)

    #tf_variables = tf.trainable_variables()
    #tf_variables = [v for v in tf_variables if "train_tower_0" and "resnet" in v.name]
    #bn_variables = [v for v in (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) if "train_tower_0" and "moving_" in v.name]

    tf_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="train_tower_0") if "resnet" in v.name]
    bn_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="train_tower_0") if "moving_" in v.name]
    fc_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="train_tower_0") if "dense" in v.name]

    print(fc_variables)
    tf_counter = 0
    tf_bn_counter = 0
    tf_fc_counter = 0

    for name in weight_names:
        if "fc" in name:
            pt_assign = reordered_weights[name]
            tf_assign = fc_variables[tf_fc_counter]
            pre_train_ops.append(tf_assign.assign(np.transpose(pt_assign)))
            tf_fc_counter += 1

        elif len(reordered_weights[name].shape) == 4:
            weight_var = reordered_weights[name]
            tf_weight = tf_variables[tf_counter]

            pre_train_ops.append(tf_weight.assign(np.transpose(weight_var, (2, 3, 1, 0))))
            tf_counter += 1

        elif "running_" in name:
            pt_assign = reordered_weights[name]
            tf_assign = bn_variables[tf_bn_counter]

            pre_train_ops.append(tf_assign.assign(pt_assign))
            tf_bn_counter += 1

        else:
            pt_assign = reordered_weights[name]
            tf_assign = tf_variables[tf_counter]

            pre_train_ops.append(tf_assign.assign(pt_assign))
            tf_counter += 1

    return tf.group(*pre_train_ops, name='load_resnet_pretrain')

def _get_init_pretrained(sess):
    saver_reader = tf.train.Saver(tf.global_variables())
    saver_reader.restore(sess, FLAGS.tune_from)

with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='image')

    with tf.variable_scope('train_tower_0') as scope:
        net = RetinaNet("resnet50")

        resnet = net.resnet(image, mode)
        gap = tf.nn.avg_pool(resnet[5], ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

        gap = tf.squeeze(gap, [1, 2])
        out = tf.layers.dense(gap, 1000, activation=None)
        out = tf.argmax(out, 1)


    #restore_model = get_init_trained()
    pretrain_op = load_pytorch_weight()
    init_op = tf.group( tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(pretrain_op)

        for n, _img in enumerate(os.listdir(img_dir)):
            img = cv2.imread(img_dir + _img)
            img = cv2.resize(img, (224, 224))

            batch_image = np.expand_dims(img, 0)
            print(batch_image.shape)
            result = sess.run(out, feed_dict={image : batch_image})

            #img = draw_boxes(img, box, label)
            #plt.imshow(img)
            print(result)
            if n==1:
                break

