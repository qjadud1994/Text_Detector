# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert VOC format dataset to TFRecord for object_detection.
For example
Hollywood head dataset:
See: http://www.di.ens.fr/willow/research/headdetection/
     Context-aware CNNs for person head detection
HDA pedestrian dataset:
See: http://vislab.isr.ist.utl.pt/hda-dataset/
Example usage:
    ./create_tf_record_pascal_fmt --data_dir=/startdt_data/HollywoodHeads2 \
        --output_dir=models/head_detector
        --label_map_path=data/head_label_map.pbtxt
        --mode=train
"""
import numpy as np
import scipy.io as sio
from random import shuffle

import io
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from lxml import etree
import PIL.Image
import tensorflow as tf
import tfrecord_utils

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/root/DB/SynthText/train/', 'Root directory to raw pet dataset, like /startdt_data/HDA_Dataset_V1.3/VOC_fmt_training_fisheye')
flags.DEFINE_string('output_dir', '/root/DB/SynthText/train/tfrecord/', 'Path to directory to output TFRecords, like models/hda_cam_person_fisheye')
FLAGS = flags.FLAGS


def dict_to_tf_example(img_path, labels):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset (here only head available) directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  if image.mode != 'RGB':
    image = image.convert('RGB')

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []

  if labels.ndim == 3:
    for i in range(labels.shape[2]):
      _xmin = float(np.min(labels[0,:,i]))
      _ymin = float(np.min(labels[1,:,i]))
      _xmax = float(np.max(labels[0,:,i]))
      _ymax = float(np.max(labels[1,:,i]))
      xmin.append(_xmin / width)
      ymin.append(_ymin / height)
      xmax.append(_xmax / width)
      ymax.append(_ymax / height)
      classes.append(1)

  else:
    _xmin = float(np.min(labels[0,:]))
    _ymin = float(np.min(labels[1,:]))
    _xmax = float(np.max(labels[0,:]))
    _ymax = float(np.max(labels[1,:]))
    xmin.append(_xmin / width)
    ymin.append(_ymin / height)
    xmax.append(_xmax / width)
    ymax.append(_ymax / height)
    classes.append(1)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': tfrecord_utils.bytes_feature(encoded_jpg),
      'image/format': tfrecord_utils.bytes_feature('jpg'.encode('utf8')),
      'image/object/bbox/xmin': tfrecord_utils.float_list_feature(xmin),
      'image/object/bbox/xmax': tfrecord_utils.float_list_feature(xmax),
      'image/object/bbox/ymin': tfrecord_utils.float_list_feature(ymin),
      'image/object/bbox/ymax': tfrecord_utils.float_list_feature(ymax),
      'image/object/class/label': tfrecord_utils.int64_list_feature(classes),
  }))
  return example


def create_tf_record(output_path, data_dir):
    """Creates a TFRecord file from examples.
    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """
    # get GT
    gt = sio.loadmat(data_dir + 'gt.mat')
    dataset_size = gt['imnames'].shape[1]
    img_files = gt['imnames'][0]
    labels = gt['wordBB'][0]

    index = list(range(dataset_size))
    shuffle(index)

    # Train_tfrecord
    writer_train = tf.python_io.TFRecordWriter(output_path + "train_2.record")

    train_index = index[0:850000]
    train_size = len(train_index)

    print ('{} training examples.', len(train_index))
    for n, i in enumerate(train_index):
        if n % 100 == 0:
            print ('On image {} of {}'.format(n, train_size), end='\r')

        img_file = data_dir + str(img_files[i][0])
        label = labels[i]

        tf_example = dict_to_tf_example(img_file, label)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()

    # Valid_tfrecord
    writer_val = tf.python_io.TFRecordWriter(output_path + "val_2.record")

    val_index = index[850000:]
    val_size = len(val_index)

    print ('{} valid examples.', val_size)
    for n, i in enumerate(val_index):
        if n % 100 == 0:
            print ('On image {} of {}'.format(n, val_size), end='\r')

        img_file = data_dir + str(img_files[i][0])
        label = labels[i]

        tf_example = dict_to_tf_example(img_file, label)
        writer_val.write(tf_example.SerializeToString())

    writer_val.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    print ("Generate data for model !")

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    # random.seed(42)
    # random.shuffle(examples_list)
    # num_examples = len(examples_list)
    # num_train = int(num_examples)
    # train_examples = examples_list[:num_train]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    create_tf_record(FLAGS.output_dir, data_dir)

if __name__ == '__main__':
  tf.app.run()
