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

"""Preprocess images and bounding boxes for detection.
We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.
A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.
The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputDataFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]
Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""

import tensorflow as tf
import random, cv2
import numpy as np

def tf_summary_image(image, boxes, name='image'):
    """Add image with bounding boxes to summary.
    """
    image = tf.expand_dims(image, 0)
    boxes = tf.expand_dims(boxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, boxes)
    tf.summary.image(name, image_with_box)
    
def normalize_image(image, mean=(0.485, 0.456, 0.406), var=(0.229, 0.224, 0.225)):
    """Normalizes pixel values in the image.
    Moves the pixel values from the current [original_minval, original_maxval]
    range to a the [target_minval, target_maxval] range.
    Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    Returns:
    image: image which is the same shape as input image.
    """
    with tf.name_scope('NormalizeImage', values=[image]):
        image = tf.to_float(image)
        image /= 255.0

        image -= mean
        image /= var

        return image


def resize_image_and_boxes(image, boxes, input_size,
                 method=tf.image.ResizeMethod.BILINEAR):
    with tf.name_scope('ResizeImage', values=[image, input_size, method]):
        image_resize = tf.image.resize_images(image, [input_size, input_size], method=method)
        boxes_resize = boxes * input_size

        return image_resize, boxes_resize


def flip_boxes_horizontally(boxes):
    """Left-right flip the boxes.
    Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    Returns:
    Horizontally flipped boxes.
    """
    # Flip boxes horizontally.
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
    
    return flipped_boxes


def flip_boxes_vertically(boxes):
    """Up-down flip the boxes
    Args:
      boxes: rank 2 float32 tensor containing bounding boxes -> [N, 4].
             Boxes are in normalized form meaning their coordinates vary
             between [0, 1]
             Each row is in the form of [ymin, xmin, ymax, xmax]
    Returns:
      Vertically flipped boxes
    """
    # Flip boxes vertically
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_ymin = tf.subtract(1.0, ymax)
    flipped_ymax = tf.subtract(1.0, ymin)
    flipped_boxes = tf.concat([flipped_ymin, xmin, flipped_ymax, xmax], axis=1)
    return flipped_boxes


def random_horizontal_flip(image, boxes, seed=None):
    """Randomly decides whether to horizontally mirror the image and detections or not.
    The probability of flipping the image is 50%.
    Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    seed: random seed
    Returns:
    image: image which is the same shape as input image.
    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.random_uniform([], seed=seed)
        # flip only if there are bounding boxes in image!
        do_a_flip_random = tf.logical_and(
            tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(
              do_a_flip_random, lambda: flip_boxes_horizontally(boxes), lambda: boxes)
            result.append(boxes)

        return tuple(result)


def random_vertical_flip(image, boxes, seed=None):
    """Randomly decides whether to vertically mirror the image and detections or not.
    The probability of flipping the image is 50%.
    Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    seed: random seed
    Returns:
    image: image which is the same shape as input image.
    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped

    with tf.name_scope('RandomVerticalFlip', values=[image, boxes]):
        result = []
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.random_uniform([], seed=seed)
        # flip only if there are bounding boxes in image!
        do_a_flip_random = tf.logical_and(
            tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
        result.append(image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(
              do_a_flip_random, lambda: flip_boxes_vertically(boxes), lambda: boxes)
            result.append(boxes)

        return tuple(result)

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
    """Scales each value in the pixels of the image.
     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.
    Returns:
    image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomPixelValueScale', values=[image]):
        color_coef = tf.random_uniform(
            tf.shape(image),
            minval=minval,
            maxval=maxval,
            dtype=tf.float32,
            seed=seed)
        image = tf.multiply(image, color_coef)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

def random_image_scale(image,
                       masks=None,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
    """Scales the image size.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
    masks: (optional) rank 3 float32 tensor containing masks with
      size [height, width, num_masks]. The value is set to None if there are no
      masks.
    min_scale_ratio: minimum scaling ratio.
    max_scale_ratio: maximum scaling ratio.
    seed: random seed.
    Returns:
    image: image which is the same rank as input image.
    masks: If masks is not none, resized masks which are the same rank as input
      masks will be returned.
    """
    with tf.name_scope('RandomImageScale', values=[image]):
        result = []
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        size_coef = tf.random_uniform([],
                                      minval=min_scale_ratio,
                                      maxval=max_scale_ratio,
                                      dtype=tf.float32, seed=seed)
        image_newysize = tf.to_int32(
            tf.multiply(tf.to_float(image_height), size_coef))
        image_newxsize = tf.to_int32(
            tf.multiply(tf.to_float(image_width), size_coef))
        image = tf.image.resize_images(
            image, [image_newysize, image_newxsize], align_corners=True)
        result.append(image)
        if masks:
            masks = tf.image.resize_nearest_neighbor(
              masks, [image_newysize, image_newxsize], align_corners=True)
            result.append(masks)
        return tuple(result)


def random_adjust_brightness(image, max_delta=32. / 255.):
    """Randomly adjusts brightness.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).
    Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
    """
    def _random_adjust_brightness(image, max_delta):
        with tf.name_scope('RandomAdjustBrightness', values=[image]):
            image = tf.image.random_brightness(image, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_brightness(image, max_delta), lambda: image)
    return image

def random_adjust_contrast(image, min_delta=0.5, max_delta=1.25):
    """Randomly adjusts contrast.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_contrast(image, min_delta, max_delta):
        with tf.name_scope('RandomAdjustContrast', values=[image]):
            image = tf.image.random_contrast(image, min_delta, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_contrast(image, min_delta, max_delta), lambda: image)
    return image

def random_adjust_hue(image, max_delta=0.02):
    """Randomly adjusts hue.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_hue(image, max_delta):
        with tf.name_scope('RandomAdjustHue', values=[image]):
            image = tf.image.random_hue(image, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image
    
    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_hue(image, max_delta), lambda: image)
    return image


def random_adjust_saturation(image, min_delta=0.5, max_delta=1.25):
    """Randomly adjusts saturation.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_saturation(image, min_delta, max_delta):
        with tf.name_scope('RandomAdjustSaturation', values=[image]):
            image = tf.image.random_saturation(image, min_delta, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image
    
    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_saturation(image, min_delta, max_delta), lambda: image)
    return image


def random_distort_color(image, color_ordering=0):
    """Randomly distorts color.
    Randomly distorts color using a combination of brightness, hue, contrast
    and saturation changes. Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).
    Returns:
    image: image which is the same shape as input image.
    Raises:
    ValueError: if color_ordering is not in {0, 1}.
    """
    with tf.name_scope('RandomDistortColor', values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            raise ValueError('color_ordering must be in {0, 1}')

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def random_jitter_boxes(boxes, ratio=0.05, seed=None):
    """Randomly jitter boxes in image.
    Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    ratio: The ratio of the box width and height that the corners can jitter.
           For example if the width is 100 pixels and ratio is 0.05,
           the corners can jitter up to 5 pixels in the x direction.
    seed: random seed.
    Returns:
    boxes: boxes which is the same shape as input boxes.
    """
    def random_jitter_box(box, ratio, seed):
        """Randomly jitter box.
        Args:
          box: bounding box [1, 1, 4].
          ratio: max ratio between jittered box and original box,
          a number between [0, 0.5].
          seed: random seed.
        Returns:
          jittered_box: jittered box.
        """
        rand_numbers = tf.random_uniform(
            [1, 1, 4], minval=-ratio, maxval=ratio, dtype=tf.float32, seed=seed)
        box_width = tf.subtract(box[0, 0, 3], box[0, 0, 1])
        box_height = tf.subtract(box[0, 0, 2], box[0, 0, 0])
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
        jittered_box = tf.add(box, hw_rand_coefs)
        jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
        return jittered_box

    with tf.name_scope('RandomJitterBoxes', values=[boxes]):
        # boxes are [N, 4]. Lets first make them [N, 1, 1, 4]
        boxes_shape = tf.shape(boxes)
        boxes = tf.expand_dims(boxes, 1)
        boxes = tf.expand_dims(boxes, 2)

        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio, seed), boxes, dtype=tf.float32)

        distorted_boxes = tf.reshape(distorted_boxes, boxes_shape)

        return distorted_boxes

    
## Random Crop

def bboxes_resize(bbox_ref, rect_boxes, quad_bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        qv = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1],bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        rect_boxes = rect_boxes - v
        quad_bboxes = quad_bboxes - qv

        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        
        qs = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        
        rect_boxes = rect_boxes / s
        quad_bboxes = quad_bboxes / qs
        
        return rect_boxes, quad_bboxes

    
def bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.
    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
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
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        #scores = tfe_math.safe_divide(inter_vol, bboxes_vol, 'intersection')
        scores = inter_vol / bboxes_vol
        return scores
    
def bboxes_filter_overlap(labels, bboxes, quad, threshold=0.4,
                          scope=None):
    """Filter out bounding boxes based on overlap with reference
    box [0, 0, 1, 1].
    Return:
      labels, bboxes: Filtered elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        quad = tf.boolean_mask(quad, mask)
        return labels, bboxes, quad
    

def distorted_bounding_box_crop(image,
                                quad_bboxes,
                                labels,
                                min_object_covered=0.05,
                                aspect_ratio_range=(0.8, 1.2),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    from utils.bbox import change_box_order
    rect_boxes = change_box_order(quad_bboxes, 'quad2yxyx')
    rect_boxes = tf.clip_by_value(rect_boxes, 0.0, 1.0)

    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, rect_boxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(rect_boxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        cropped_rect_boxes, cropped_quad_bboxes = bboxes_resize(distort_bbox, rect_boxes, quad_bboxes)
        cropped_labels, cropped_rect_boxes, cropped_quad_bboxes = bboxes_filter_overlap(labels, cropped_rect_boxes, cropped_quad_bboxes)

        no_box = tf.equal(tf.shape(cropped_quad_bboxes)[0], 0) # If there is no box in the image, it returns the original image.
        image, quad_bboxes, labels = tf.cond(no_box, lambda:(image, quad_bboxes, labels), lambda:(cropped_image, cropped_quad_bboxes, cropped_labels))
        
        quad_bboxes = tf.clip_by_value(quad_bboxes, 0.0, 1.0)

        return cropped_image, quad_bboxes, labels
    
    
def rotate(image, boxes):
    height, width, _ = image.shape
    center = (width/2, height/2)
    size = (width, height)
    
    deg = 90
    rotate_prob = 3
    mean = (104, 117, 123)
    
    if random.randint(0, rotate_prob) == 0:
        angle = random.randint(-deg, deg)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        image = cv2.warpAffine(image, M, size, borderValue=mean)
        
        for k, box in enumerate(boxes):
            for l in range(4):
                pt = np.append(box[2*l:2*(l+1)], 1)
                rot_pt = M.dot(pt).transpose()
                boxes[k,2*l:2*(l+1)] = rot_pt[:2]

    return image, boxes
