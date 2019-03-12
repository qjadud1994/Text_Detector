import os
import tensorflow as tf

slim = tf.contrib.slim

class InputProducer(object):

    def __init__(self, preprocess_image_fn=None, vertical_image=False):
        self.vertical_image = vertical_image
        self._preprocess_image = preprocess_image_fn if preprocess_image_fn is not None \
                                 else self._default_preprocess_image_fn

        self.ITEMS_TO_DESCRIPTIONS = {
            'image': 'A color image of varying height and width.',
            'shape': 'Shape of the image',
            'object/bbox': 'A list of bounding boxes, one per each object.',
            'object/label': 'A list of labels, one per each object.',
        }

        self.SPLITS_TO_SIZES = {
            'train_IC13': 229,
            'val_IC13': 233,
            'train_2': 850000,
            'val_2': 8750,
            'train_quad': 850000,
            'val_quad': 8750,
            'train_IC15': 1000,
            'val_IC15': 500,
            'train_IC15_mask': 1000,
            'val_IC15_mask': 500
        }

        self.FILE_PATTERN = '%s.record'

    def num_classes(self):
        return 20

    def get_split(self, split_name, dataset_dir, is_rect=True):
        """Gets a dataset tuple with instructions for reading Pascal VOC dataset.
        Args:
          split_name: A train/test split name.
          dataset_dir: The base directory of the dataset sources.
          file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
          reader: The TensorFlow reader type.
        Returns:
          A `Dataset` namedtuple.
        Raises:
            ValueError: if `split_name` is not a valid train/test split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        file_pattern = os.path.join(dataset_dir, self.FILE_PATTERN % split_name)

        reader = tf.TFRecordReader

        if is_rect:  # Rect annotations
            keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            }
            items_to_handlers = {
                'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
                'object/bbox': slim.tfexample_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
                'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
            }
            
        else: #Quad annotations
            keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/object/bbox/y0': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/x0': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
                'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            }
            items_to_handlers = {
                'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
                'object/quad1': slim.tfexample_decoder.BoundingBox(
                    ['y0', 'x0', 'y1', 'x1'], 'image/object/bbox/'),
                'object/quad2': slim.tfexample_decoder.BoundingBox(
                    ['y2', 'x2', 'y3', 'x3'], 'image/object/bbox/'),
                'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
            }
            
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        #if has_labels(dataset_dir):
        #    labels_to_names = read_label_file(dataset_dir)

        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=self.SPLITS_TO_SIZES[split_name],
            items_to_descriptions=self.ITEMS_TO_DESCRIPTIONS,
            num_classes=self.num_classes(),
            labels_to_names=labels_to_names)

    def _default_preprocess_image_fn(self, image, is_train=True):
        return image
