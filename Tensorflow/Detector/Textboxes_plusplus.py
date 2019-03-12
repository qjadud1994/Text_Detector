import numpy as np
import collections

from Detector.layers import *
from utils.bbox import iou, change_box_order, box_iou
from utils.preprocess import *

#from Detector import Network
from tensorflow.contrib import learn
from Detector.input_producer import InputProducer

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
resnet_version = {"resnet50": [3, 4, 6, 3],
                  "resnet101": [3, 4, 23, 3],
                  "resnet152": [3, 8, 36, 3],
                  "se-resnet50": [3, 4, 6, 3],
                  "se-resnet101": [3, 4, 23, 3]}

class RetinaNet():
    def __init__(self, backbone, loss_fn=None):
        #super().__init__(out_charset)

        # Set tune scope
        self.scope="resnet_model|FPN|head"

        assert backbone in resnet_version.keys()
        self.backbone = backbone

        self.use_se_block = "se-resnet" in backbone

        self.input_size = FLAGS.input_size
        self.input_shape = np.array([self.input_size, self.input_size])

        self.use_bn = FLAGS.use_bn
        self.probability = 0.01
        self.cls_thresh = FLAGS.cls_thresh
        self.nms_thresh = FLAGS.nms_thresh
        self.max_detect = FLAGS.max_detect

        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        if FLAGS.test in ["v5"]:
            self.anchor_areas = [16*16., 32*32., 64*64., 128*128., 256*256., 512*512.]  # p2 -> p7
        elif FLAGS.test in ["v6"]:
            self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512., 1024*1024.]  # p2 -> p7
        
        self.aspect_ratios = [1., 2., 3., 5., 1./2., 1./3., 1./5.]

        self.num_anchors = len(self.aspect_ratios)* 2 #* len(self.scale_ratios)
        self.anchor_rect_boxes = self._get_anchor_boxes()
        self.anchor_quad_boxes = change_box_order(self.anchor_rect_boxes, "yxhw2quad")

        #print("backbone : ", self.backbone)
        #print("use_bn : ", self.use_bn)
        #print("use_se_block : ", self.use_se_block)
        #print("input_size : ", self.input_size)
        #print("experiment model : ", FLAGS.test)

    def preprocess_image(self, image, boxes, labels, is_train=True):
        """ pre-process / Augmentation """
        if is_train:
            image, boxes, labels = distorted_bounding_box_crop(image, boxes, labels)

            #image, boxes = random_horizontal_flip(image, boxes)
            #image, boxes = random_vertical_flip(image, boxes)

            image, boxes = resize_image_and_boxes(image, boxes, self.input_size)
            image = normalize_image(image)

            image = random_adjust_brightness(image)
            image = random_adjust_contrast(image)
            image = random_adjust_hue(image)
            image = random_adjust_saturation(image)

        else:
            #image, boxes, labels = distorted_bounding_box_crop(image, boxes, labels)

            image, boxes = resize_image_and_boxes(image, boxes, self.input_size)
            image = normalize_image(image)

        return image, boxes, labels

    def get_logits(self, inputs, mode, **kwargs):
        """Get RetinaNet logits(output)"""
        features_resnet = self.resnet(inputs, mode, self.use_bn)
        features = self.FPN(features_resnet, mode)

        with tf.variable_scope("head"):
            box_subnet = []
            class_subnet = []
            for n, feature in enumerate(features):
                if FLAGS.test in ["v1", "v2"]:
                    _box = self.head(feature, self.num_anchors * (4 + 8), "C%d_loc_head" % (n+3)) #4d-rect + 8d-quad
                    _box = tf.reshape(_box, [FLAGS.batch_size, -1, 4 + 8])
                elif FLAGS.test in ["v3", "v4", "v5"]:
                    _box = self.head(feature, self.num_anchors * 8, "C%d_loc_head" % (n+3)) #4d-rect + 8d-quad
                    _box = tf.reshape(_box, [FLAGS.batch_size, -1, 8])
                
                if FLAGS.test in ["v1", "v3"]:
                    _class = self.head(feature, self.num_anchors * 2, "C%d_cls_head" % (n+3))
                    _class = tf.reshape(_class, [FLAGS.batch_size, -1, 2])
                elif FLAGS.test in ["v2", "v4", "v5"]:
                    _class = self.head(feature, self.num_anchors * 1, "C%d_cls_head" % (n+3))
                    _class = tf.reshape(_class, [FLAGS.batch_size, -1, 1])

                box_subnet.append(_box)
                class_subnet.append(_class)

            logits = tf.concat(box_subnet, axis=1), tf.concat(class_subnet, axis=1)

            return logits


    def resnet(self, inputs, mode, use_bn):
        """Build convolutional network layers attached to the given input tensor"""
        training = (mode == learn.ModeKeys.TRAIN) and not FLAGS.bn_freeze

        blocks = resnet_version[self.backbone]

        with tf.variable_scope("resnet_model"):
            ## stage 1
            C1 = conv_layer(inputs, 64, kernel_size=7, strides=2)
            C1 = norm_layer(C1, training, use_bn)
            C1 = pool_layer(C1, (3, 3), stride=(2, 2))

            ## stage2
            C2 = res_block(C1, [64, 64, 256], training, use_bn, self.use_se_block, strides=1, downsample=True)
            for i in range(blocks[0] - 1):
                C2 = res_block(C2, [64, 64, 256], training, use_bn, self.use_se_block)

            ## stage3
            C3 = res_block(C2, [128, 128, 512], training, use_bn, self.use_se_block, strides=2, downsample=True)
            for i in range(blocks[1] - 1):
                C3 = res_block(C3, [128, 128, 512], training, use_bn, self.use_se_block)

            ## stage4
            C4 = res_block(C3, [256, 256, 1024], training, use_bn, self.use_se_block, strides=2, downsample=True)
            for i in range(blocks[2] - 1):
                C4 = res_block(C4, [256, 256, 1024], training, use_bn, self.use_se_block)

            ## stage5
            C5 = res_block(C4, [512, 512, 2048], training, use_bn, self.use_se_block, strides=2, downsample=True)
            for i in range(blocks[3] - 1):
                C5 = res_block(C5, [512, 512, 2048], training, use_bn, self.use_se_block)

            return [None, C1, C2, C3, C4, C5]

    def FPN(self, C, mode):

        with tf.variable_scope("FPN"):  #TO do... check FPN for ReinaNet
            P5 = conv_layer(C[5], 256, kernel_size=1)
            
            P4 = upsampling(P5, size=(2, 2)) + conv_layer(C[4], 256, kernel_size=1)
            P4 = conv_layer(P4, 256, kernel_size=3)

            P3 = upsampling(P4, size=(2, 2)) + conv_layer(C[3], 256, kernel_size=1)
            P3 = conv_layer(P3, 256, kernel_size=3)

            P6 = conv_layer(C[5], 256, kernel_size=3, strides=2)
            P7 = relu(P6)
            P7 = conv_layer(P7, 256, kernel_size=3, strides=2)

            if FLAGS.test in ["v5"]:
                P2 = upsampling(P3, size=(2, 2)) + conv_layer(C[2], 256, kernel_size=1)
                P2 = conv_layer(P2, 256, kernel_size=3)
                return P2, P3, P4, P5, P6, P7
            
        return P3, P4, P5, P6, P7

    def head(self, feature, out, scope):
        with tf.variable_scope(scope):
            _kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            for _ in range(4):
                feature = conv_layer(feature, 256, kernel_size=3, use_bias=False, kernel_initializer=_kernel_initializer)
                feature = relu(feature)

            if "cls" in scope: #cls_subnet
                feature = conv_layer(feature, out, kernel_size=(3, 5), kernel_initializer=_kernel_initializer)
                bias_initial = tf.ones(out, dtype=tf.float32) * -tf.log((1 - self.probability) / self.probability)
                feature = tf.nn.bias_add(feature, bias_initial)

            elif "loc" in scope: #loc_subnet
                feature = conv_layer(feature, out, kernel_size=(3, 5), kernel_initializer=_kernel_initializer)
            return feature

    def get_loss(self, y_pred, y_true, alpha=0.25, gamma=2.0):

        def regression_loss(pred_boxes, gt_boxes, weights=1.0):
            """Regression loss (Smooth L1 loss (=huber loss))
                    pred_boxes: [# anchors, 4]
                    gt_boxes: [# anchors, 4]
                    weights: Tensor of weights multiplied by loss with shape [# anchors]
            """            
            #loc_loss = tf.losses.huber_loss(labels=gt_boxes, predictions=pred_boxes,
            #weights=weights, scope='box_loss')
            #return loc_loss
            x = tf.abs(pred_boxes-gt_boxes)
            x = tf.where(tf.less(x, 1.0), 0.5*x**2, x-0.5)
            x = tf.reduce_sum(x)
            return x

        def focal_loss(preds_cls, gt_cls,
                       alpha=0.25, gamma=2.0, name=None, scope=None):
            """Compute sigmoid focal loss between logits and onehot labels"""

            #with tf.name_scope(scope, 'focal_loss', [preds_cls_onehot, gt_cls_onehot]) as sc:
            if FLAGS.test  in ["v1", "v3"]:
                gt_cls = tf.one_hot(indices=gt_cls - 1, depth=2, dtype=tf.float32)
            elif FLAGS.test  in ["v2", "v4", "v5"]:
                gt_cls = tf.one_hot(indices=gt_cls - 1, depth=1, dtype=tf.float32)
            #gt_cls = tf.one_hot(gt_cls, FLAGS.num_classes+1, dtype=tf.float32)
            #gt_cls = gt_cls[:, 1:]

            preds_cls = tf.nn.sigmoid(preds_cls)
            
            # cross-entropy -> if y=1 : pt=p / otherwise : pt=1-p
            predictions_pt = tf.where(tf.equal(gt_cls, 1.0), preds_cls, 1.0 - preds_cls)

            # add small value to avoid 0
            epsilon = 1e-8
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(predictions_pt, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(gt_cls, 1.0), alpha_t, 1.0 - alpha_t)
            gamma_t = tf.scalar_mul(gamma, tf.ones_like(predictions_pt, tf.float32))

            focal_losses = alpha_t * (-tf.pow(1.0 - predictions_pt, gamma_t) * tf.log(predictions_pt))
            #focal_losses = alpha_t * tf.pow(1. - predictions_pt, gamma) * -tf.log(predictions_pt + epsilon)
            focal_losses = tf.reduce_sum(focal_losses, axis=1)
            return focal_losses

        loc_preds, cls_preds = y_pred
        loc_gt, cls_gt = y_true

        # number of positive anchors
        valid_anchor_indices = tf.where(tf.greater(cls_gt, 0))
        gt_anchor_nums = tf.shape(valid_anchor_indices)[0]

        """Location Regression loss"""
        # skip negative and ignored anchors
        valid_loc_preds = tf.gather_nd(loc_preds, valid_anchor_indices)
        valid_loc_gt = tf.gather_nd(loc_gt, valid_anchor_indices)

        loc_loss = regression_loss(valid_loc_preds, valid_loc_gt)
        loc_loss = tf.truediv(tf.reduce_sum(loc_loss), tf.to_float(gt_anchor_nums))
        
        if FLAGS.test in ["v1", "v2"]:
            loc_loss *= 0.2
        elif FLAGS.test in ["v3" ,"v4", "v5"]:
            loc_loss *= 0.5

        """Classification loss"""
        valid_cls_indices = tf.where(tf.greater(cls_gt, -1))

        # skip ignored anchors (iou belong to 0.4 to 0.5)
        valid_cls_preds = tf.gather_nd(cls_preds, valid_cls_indices)
        valid_cls_gt = tf.gather_nd(cls_gt, valid_cls_indices)

        cls_loss = focal_loss(valid_cls_preds, valid_cls_gt)
        cls_loss = tf.truediv(tf.reduce_sum(cls_loss), tf.to_float(gt_anchor_nums))

        """Variables"""
        scope = self.scope or FLAGS.tune_scope
        scope = '|'.join(['train_tower_[0-9]+/' + s for s in scope.split('|')])

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return loc_loss, cls_loss, tvars, extra_update_ops

    def _get_anchor_hw(self):

        anchor_hw = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                anchor_h = np.sqrt(s/ar)
                anchor_w = ar * anchor_h
                anchor_hw.append([anchor_h, anchor_w])

        num_fms = len(self.anchor_areas)
        anchor_hw = np.array(anchor_hw)
        return anchor_hw.reshape(num_fms, -1, 2)

    def _get_anchor_boxes(self):
        anchor_hw = self._get_anchor_hw()
        num_fms = len(self.anchor_areas)
        
        if FLAGS.test in ["v5"]:
            fm_sizes = [np.ceil(self.input_shape/pow(2.,i+2)) for i in range(num_fms)]  # p3 -> p7 feature map sizes
        else:
            fm_sizes = [np.ceil(self.input_shape/pow(2.,i+3)) for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = self.input_shape / fm_size
            
            fm_h, fm_w = int(fm_size[0]), int(fm_size[1]) # fm_h == fm_w : True
            fm_w = fm_w * 2 #- 1  #for add vertical offset

            meshgrid_x = ((np.arange(0, fm_w) / 2) + 0.5) * grid_size[0]
            meshgrid_y = (np.arange(0, fm_h) + 0.5) * grid_size[1]
            meshgrid_x, meshgrid_y = np.meshgrid(meshgrid_x, meshgrid_y)

            yx = np.vstack((meshgrid_y.ravel(), meshgrid_x.ravel())).transpose()
            yx = np.tile(yx.reshape((fm_h, fm_w, 1, 2)), (len(self.aspect_ratios), 1))

            hw = np.tile(anchor_hw[i].reshape(1, 1, len(self.aspect_ratios), 2), (fm_h, fm_w, 1, 1))
            box = np.concatenate([yx, hw], 3)  # [y,x,h,w]
            boxes.append(box.reshape(-1, 4))

        return tf.cast(tf.concat(boxes, 0), tf.float32)

    def encode(self, gt_quad_boxes, labels):
        """gt_quad_boxes : yxyx , anchor_rect_boxes : yxhw"""
        
        #anchor_quad_boxes = change_box_order(self.anchor_rect_boxes, "yxhw2quad")
        gt_rect_boxes = change_box_order(gt_quad_boxes, "quad2yxyx")

        ious = iou(self.anchor_rect_boxes, gt_rect_boxes) #(num_anchor, num_gt_box)

        max_ids = tf.argmax(ious, axis=1, name="encode_argmax")
        max_ious = tf.reduce_max(ious, axis=1)

        #Each anchor box matches the largest iou with the gt box
        gt_quad_boxes = tf.gather(gt_quad_boxes, max_ids) #(num_anchor, 8)
        gt_rect_boxes = tf.gather(gt_rect_boxes, max_ids) #(num_anchor, 4)
        
        # for Rectangle boxes
        gt_rect_boxes = change_box_order(gt_rect_boxes, "yxyx2yxhw")
        loc_rect_yx = (gt_rect_boxes[:, :2] - self.anchor_rect_boxes[:, :2]) / self.anchor_rect_boxes[:, 2:]
        loc_rect_hw = tf.log(gt_rect_boxes[:, 2:] / self.anchor_rect_boxes[:, 2:])

        # for Quad boxes
        anchor_boxes_hw = tf.tile(self.anchor_rect_boxes[:, 2:4], [1, 4])
        loc_quad_yx = (gt_quad_boxes - self.anchor_quad_boxes) / anchor_boxes_hw
        
        if FLAGS.test  in ["v1", "v2"]:
            loc_targets = tf.concat([loc_rect_yx, loc_rect_hw, loc_quad_yx], 1) # (num_anchor, 12)
        elif FLAGS.test  in ["v3", "v4", "v5"]:
            #loc_targets = tf.concat([loc_quad_yx], 1) # (num_anchor, 12)
            loc_targets = loc_quad_yx
        
        # TODO: check cls
        if FLAGS.test in ["v1", "v3"]:
            cls_targets = 1 + tf.gather(labels, max_ids)    # text : 1  
        elif FLAGS.test in ["v2", "v4", "v5"]:
            cls_targets = tf.gather(labels, max_ids)    # text : 1
        
        # iou < 0.4 : background(0)  /  0.4 < iou < 0.5 : ignore(-1)
        cls_targets = tf.where(tf.less(max_ious, 0.5), -tf.ones_like(cls_targets), cls_targets)
        cls_targets = tf.where(tf.less(max_ious, 0.4), tf.zeros_like(cls_targets), cls_targets)

        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds):
        if len(loc_preds.shape.as_list()) == 3:
            loc_preds = tf.squeeze(loc_preds, 0)
            cls_preds = tf.squeeze(cls_preds, 0)

        if loc_preds.dtype != tf.float32:
            loc_preds = tf.cast(loc_preds, tf.float32)

        if FLAGS.test in ["v1", "v2"]:
            loc_yx = loc_preds[:, :2]
            loc_hw = loc_preds[:, 2:4]

            loc_quad = loc_preds[:, 4:]

            yx = loc_yx * self.anchor_rect_boxes[:, 2:4] + self.anchor_rect_boxes[:, 0:2]
            hw = tf.exp(loc_hw) * self.anchor_rect_boxes[:, 2:4]

            boxes = tf.concat([yx-hw/2, yx+hw/2], axis=1)  # [#anchors,4], yxyx
            boxes = tf.clip_by_value(boxes, 0, self.input_size)
            
        elif FLAGS.test in ["v3", "v4", "v5"]:
            loc_quad = loc_preds
            
        quad_boxes = self.anchor_quad_boxes + tf.tile(self.anchor_rect_boxes[:, 2:4], [1, 4]) * loc_quad  # [#anchor, 8]
        quad_boxes = tf.clip_by_value(quad_boxes, 0, self.input_size)

        cls_preds = tf.nn.sigmoid(cls_preds)
        labels = tf.argmax(cls_preds, axis=1, name="decode_argmax")
        score = tf.reduce_max(cls_preds, axis=1)

        ids = tf.where(score > self.cls_thresh)
        ids = tf.squeeze(ids, axis=1)
        
        score = tf.gather(score, ids)
        labels = tf.gather(labels, ids)
        quad_boxes = tf.gather(quad_boxes, ids)

        if FLAGS.test in ["v1" ,"v2"]:
            boxes = tf.gather(boxes, ids)
            keep = tf.image.non_max_suppression(boxes, score, self.max_detect, self.nms_thresh)

            rect_output = {'boxes' : tf.gather(boxes, keep),
                           'labels' : tf.gather(labels, keep),
                           'score' : tf.gather(score, keep)}
        
            return rect_output, labels, score, quad_boxes
        
        elif FLAGS.test in ["v3" ,"v4", "v5"]:
            return labels, score, quad_boxes

    def get_input(self,
                  is_train=True,
                  num_gpus=1):
        input_features = []

        InputFeatures = collections.namedtuple('InputFeatures', ('image', 'loc', 'cls'))
        input_producer = InputProducer()
        for gpu_indx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_indx):
                if is_train:
                    split_name = 'train_quad' if "SynthText" in FLAGS.train_path else 'train_IC15'
                    batch_size = FLAGS.batch_size
                else:
                    split_name = 'val_quad' if "SynthText" in FLAGS.train_path else 'val_IC15'
                    batch_size = FLAGS.valid_batch_size

                dataset = input_producer.get_split(split_name, FLAGS.train_path, is_rect=False)

                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_input_threads,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 * batch_size,
                    shuffle=True)
                _images, _quad1, _quad2, _labels = provider.get(['image', 'object/quad1', 'object/quad2', 'object/label'])
                _bboxes = tf.concat([_quad1, _quad2], axis=1) # 8-points QUAD
                
                # pre-processing & encode
                _images, _bboxes, _labels = self.preprocess_image(_images, _bboxes, _labels, is_train)

                _bboxes, _labels = self.encode(_bboxes, _labels)

                images, bboxes, labels = tf.train.shuffle_batch(
                    [_images, _bboxes, _labels],
                    batch_size=batch_size,
                    num_threads=FLAGS.num_input_threads,
                    capacity=20*batch_size,
                    min_after_dequeue=10*batch_size)

                input_features.append(InputFeatures(images, bboxes, labels))
        return input_features

    '''
    def get_test_input(self, batch_size):
        InputFeatures = collections.namedtuple('InputFeatures', ('image', 'bboxes', 'labels'))
        image = tf.placeholder(tf.uint8,
                shape=[batch_size, FLAGS.input_size, FLAGS.input_size, 3], name='image')
        bboxes = tf.placeholder(tf.float32, shape=[batch_size, 4], name='bbox')
        labels = tf.placeholder(tf.int32, shape=[batch_size], name='labels')

        input_feature = InputFeatures(image, bboxes, labels)
        return input_feature
    '''
