import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import cv2, os, time
import zipfile
#os.environ["CUDA_VISIBLE_DEVICES"]="4"

from PIL import Image, ImageDraw
from tensorflow.contrib import learn
from Detector.Textboxes_plusplus import RetinaNet
from utils.nms_poly import non_max_suppression_poly

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.WARN)
tf.app.flags.DEFINE_string('f', '', 'kernel')
#### Input pipeline
tf.app.flags.DEFINE_string('backbone', "se-resnet50",
                            """select RetinaNet backbone""")
tf.app.flags.DEFINE_integer('input_size', 608,
                            """Input size""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Train batch size""")
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """number of classes""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """The number of gpu""")
tf.app.flags.DEFINE_string('test', '', """Training mode""")
#tf.app.flags.DEFINE_string('tune_from', 'logs_synth_se/bn_momen2/model.ckpt-100000',
#                          """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_from', 'logs_synth_se_plus/bn_adam1/best_models/model-85400',
                         """Path to pre-trained model checkpoint""")

#### Training config
tf.app.flags.DEFINE_boolean('use_bn', True,
                            """use batchNorm or GroupNorm""")
tf.app.flags.DEFINE_float('cls_thresh', 0.3,
                            """thresh for class""")
tf.app.flags.DEFINE_float('nms_thresh', 0.4,
                            """thresh for nms""")
tf.app.flags.DEFINE_integer('max_detect', 300,
                            """num of max detect (using in nms)""")
tf.app.flags.DEFINE_string('output_zip', "IC15_result1.zip",
                            """output zip file name""")
tf.app.flags.DEFINE_boolean('save_image', False,
                            """output zip file name""")

# test image path & list
img_dir = "/root/DB/ICDAR2015_Incidental/test/"
val_list = [im for im in os.listdir(img_dir) if "jpg" in im]

if not os.path.exists("result/"):
    os.mkdir("result/")
    
# save results dir & zip
eval_dir = "/root/Detector/ocr_evaluation/code/icdar/4_incidental_scene_text/1_TextLocalization/1_IoU/"
result_zip = zipfile.ZipFile(eval_dir + FLAGS.output_zip, 'w')

#
time_list = []
mode = learn.ModeKeys.INFER

def _get_init_pretrained(sess):
    saver_reader = tf.train.Saver(tf.global_variables())
    saver_reader.restore(sess, FLAGS.tune_from)
    
with tf.Graph().as_default():
    _image = tf.placeholder(tf.float32, shape=[None, None, 3], name='image')

    with tf.variable_scope('train_tower_0') as scope:
        net = RetinaNet(FLAGS.backbone)
        
        image = tf.expand_dims(_image, 0)
        image = tf.to_float(image)
        image /= 255.0
        
        mean = (0.485, 0.456, 0.406)
        var = (0.229, 0.224, 0.225)
        
        image -= mean
        image /= var
        
        image = tf.image.resize_images(image, (FLAGS.input_size, FLAGS.input_size),
                                           method=tf.image.ResizeMethod.BILINEAR)
        
        box_head, cls_head = net.get_logits(image, mode)

        decode = net.decode(box_head, cls_head)

    init_op = tf.group( tf.global_variables_initializer(),
                        tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        _get_init_pretrained(sess)
        
        for n, _img in enumerate(val_list):
            save_file = "res_%s.txt" % (_img[:-4])
            #print("save [%d/%d] %s" % (n, len(val_list), save_file), end='\r')
            f = open("result/res_%s.txt" % (_img[:-4]), "w")
            
            img = Image.open(img_dir + _img)
            width, height = img.size
            
            #start = time.time()
            #boxes, labels, scores = sess.run(decode, feed_dict={_image : img})
            if FLAGS.test in ["v1" ,"v2"]:
                rect_output, labels, scores, quad_boxes = sess.run(decode, feed_dict={_image : img})
                #time_list.append(time.time() - start)
            
                rect_output['boxes'] /= FLAGS.input_size
                rect_output['boxes'] *= ([height, width] * 2)
                #img = draw_text_boxes(img, rect_output['boxes'])
            
            elif FLAGS.test in ["v3" ,"v4"]:
                labels, scores, quad_boxes = sess.run(decode, feed_dict={_image : img})
                
            quad_boxes /= FLAGS.input_size
            quad_boxes *= ([height, width] * 4)
            
            quad_boxes = quad_boxes.reshape((-1, 4, 2)).astype(np.int32)
            quad_boxes = quad_boxes[:, :, ::-1]
            
            nms_indices = non_max_suppression_poly(quad_boxes, scores, FLAGS.nms_thresh)
            quad_boxes = quad_boxes[nms_indices]
            
            for quad in quad_boxes:
                [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
                f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))
                             
            f.close()
            result_zip.write( "result/" + save_file, save_file, compress_type=zipfile.ZIP_DEFLATED)
            os.remove("result/res_%s.txt" % (_img[:-4]))
            
            if FLAGS.save_image:
                from utils.bbox import draw_text_boxes
                save_img_dir = "/home/beom/samba/textboxes_result/"
                gt = open("result/gt/gt_%s.txt" % (_img[:-4]), "r")
                gt = gt.readlines()
                gt_boxes = []

                for g in gt:
                    xmin, ymin, xmax, ymax, label = g.split(",")[:5]
                    
                    if "###" not in label:
                        gt_boxes.append([ymin, xmin, ymax, xmax])
                
                img = draw_text_boxes(img, gt_boxes, color=(255,0,0))
                img = draw_text_boxes(img, boxes, color=(0,255,0))
                img.save(save_img_dir + _img[:-4] + ".jpg")
                

#time_list = time_list[1:]
#avg_time = sum(time_list) / len(time_list)
#print("\navg FPS = %.3f" % (1 / avg_time))
result_zip.close()

import subprocess
query = "python2 %sscript.py -g=%sgt.zip -s=%s" % (eval_dir, eval_dir, eval_dir+FLAGS.output_zip)
# return value
subprocess.call(query, shell=True)
# scorestring = subprocess.check_output(query, shell=True)
os.remove(eval_dir+FLAGS.output_zip)

#print("\n\n========== result [ %s ] ==== / option [ input_size=%d, cls_thresh=%.2f, nms_thresh=%.2f======\n" % (FLAGS.tune_from, FLAGS.input_size, FLAGS.cls_thresh, FLAGS.nms_thresh ))
