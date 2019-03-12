import importlib
import os, sys
import numpy as np
import tensorflow as tf
import collections
from pprint import pprint
from subprocess import Popen, PIPE
from tensorflow.contrib import learn

from Detector.Textboxes_plusplus import RetinaNet
from utils.bbox import draw_bboxes
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

#### Input pipeline
tf.app.flags.DEFINE_string('backbone', "se-resnet50",
                            """select RetinaNet backbone""")
tf.app.flags.DEFINE_integer('input_size', 608,
                            """Input size""")
tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Train batch size""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Learninig rate""")
tf.app.flags.DEFINE_integer('num_input_threads', 2,
                            """Number of readers for input data""")
tf.app.flags.DEFINE_string('test', 'v5', """Training mode""")

#### Train dataset
# /root/DB/SynthText/train/tfrecord/  or  /root/DB/ICDAR2015_Incidental/tfrecord/
tf.app.flags.DEFINE_string('train_path', '', 
                           """Base directory for training data""")

### Validation dataset (during training)
tf.app.flags.DEFINE_string('valid_dataset','validation',
                          """Validation dataset name""")
tf.app.flags.DEFINE_integer('valid_device', 0,
                           """Device for validation""")
tf.app.flags.DEFINE_integer('valid_batch_size', 8,
                            """Validation batch size""")
tf.app.flags.DEFINE_boolean('use_validation', False,
                            """Whether use validation or not""")
tf.app.flags.DEFINE_integer('valid_steps', 250,
                            """Validation steps""")
tf.app.flags.DEFINE_boolean('use_evaluation', True,
                            """Whether use evaluation or not""")

#### Output Path
tf.app.flags.DEFINE_string('output', 'logs_IC15_se_plus/bn_momentum1',
                           """Directory for event logs and checkpoints""")
#### Training config
tf.app.flags.DEFINE_boolean('use_bn', True,
                            """use batchNorm or GroupNorm""")
tf.app.flags.DEFINE_boolean('bn_freeze', True,
                            """Freeze batchNorm or not""")
tf.app.flags.DEFINE_float('cls_thresh', 0.3,
                            """thresh for class""")
tf.app.flags.DEFINE_float('nms_thresh', 0.25,
                            """thresh for nms""")
tf.app.flags.DEFINE_integer('max_detect', 300,
                            """num of max detect (using in nms)""")
tf.app.flags.DEFINE_string('tune_from', '',
                           """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope', '',
                           """Variable scope for training""")
tf.app.flags.DEFINE_integer('max_num_steps', 120000,
                            """Number of optimization steps to run""")
tf.app.flags.DEFINE_boolean('verbose', False,
                            """Print log in tensorboard""")
tf.app.flags.DEFINE_boolean('use_profile', False,
                            """Whether use Tensorflow Profiling""")
tf.app.flags.DEFINE_boolean('use_debug', False,
                            """Whether use TFDBG or not""")
tf.app.flags.DEFINE_integer('save_steps', 250,
                            """Save steps""")
tf.app.flags.DEFINE_integer('summary_steps', 100,
                            """Save steps""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """Moving Average dacay factor""")
tf.app.flags.DEFINE_float('weight_decay', 1e-4,
                            """weight dacay factor""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                            """momentum factor""")


mode = learn.ModeKeys.TRAIN

TowerResult = collections.namedtuple('TowerResult', ('tvars',
                                                     'loc_loss', 'cls_loss',
                                                     'grads', 'extra_update_ops',
                                                     'optimizer'))

ValidTowerResult = collections.namedtuple('ValidTowerResult', ('loc_loss', 'cls_loss'))

def _get_session(monitored_sess):
    session = monitored_sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session


def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not FLAGS.tune_from:
        return None
    saver_reader = tf.train.Saver(tf.global_variables())
    model_path = FLAGS.tune_from

    def init_fn(scaffold, sess): 
        return saver_reader.restore(sess, model_path)
    return init_fn


def _average_gradients(tower_grads):
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grads = tf.stack([g for g, _ in grads_and_vars])
        grad = tf.reduce_mean(grads, 0)
        v = grads_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def allreduce_grads(all_grads, average=True):
    from tensorflow.contrib import nccl
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads_and_vars in zip(*all_grads):
        grads = [g for g, _ in grads_and_vars]
        _vars = [v for _, v in grads_and_vars]
        summed = nccl.all_sum(grads)
        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower, name='allreduce_avg')
            grads_for_devices.append(g)
        new_all_grads.append(zip(grads_for_devices, _vars))

    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret

def _get_post_init_ops():
    """
    Copy values of variables on GPU 0 to other GPUs.
    """
    # literally all variables, because it's better to sync optimizer-internal variables as well
    all_vars = tf.global_variables() + tf.local_variables()
    var_by_name = dict([(v.name, v) for v in all_vars])
    post_init_ops = []
    for v in all_vars:
        if not v.name.find('tower') >= 0:
            continue
        if v.name.startswith('train_tower_0'):
            # no need for copy to tower0
            continue
        # in this trainer, the master name doesn't have the towerx/ prefix
        split_name = v.name.split('/')
        prefix = split_name[0]
        realname = '/'.join(split_name[1:])
        if prefix in realname:
            # logger.warning("variable {} has its prefix {} appears multiple times in its name!".format(v.name, prefix))
            pass
        copy_from = var_by_name.get(v.name.replace(prefix, 'train_tower_0'))
        if copy_from is not None:
            post_init_ops.append(v.assign(copy_from.read_value()))
        else:
            # logger.warning("Cannot find {} in the graph!".format(realname))
            pass
    # logger.info("'sync_variables_from_main_tower' includes {} operations.".format(len(post_init_ops)))
    return tf.group(*post_init_ops, name='sync_variables_from_main_tower')


def load_pytorch_weight(use_bn, use_se_block):
    from torch import load

    if use_bn:
        if use_se_block:
            pt_load = load("weights/se_resnet50-ce0d4300.pth")
        else:
            pt_load = load("weights/resnet50.pth")
    else:
        pt_load = load("weights/resnet50_groupnorm32.tar")['state_dict']
    reordered_weights = {}
    pre_train_ops = []

    for key, value in pt_load.items():
        try:
            reordered_weights[key] = value.data.cpu().numpy()
        except:
            reordered_weights[key] = value.cpu().numpy()

    weight_names = list(reordered_weights)

    tf_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="train_tower_0/resnet_model")]

    if use_bn:   # BatchNorm
        bn_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="train_tower_0/resnet_model") if
                        "moving_" in v.name]
        tf_counter = 0
        tf_bn_counter = 0

        for name in weight_names:
            if not use_se_block and "fc" in name:    # last fc layer (resnet)
                continue
            if use_se_block and "last_linear" in name:  # last fc layer(se-resnet)
                continue

            elif len(reordered_weights[name].shape) == 4:
                if "se_module" in name: #se_block
                    pt_assign = np.squeeze(reordered_weights[name])
                    tf_assign = tf_variables[tf_counter]

                    pre_train_ops.append(tf_assign.assign(np.transpose(pt_assign)))
                    tf_counter += 1
                else: #conv
                    weight_var = reordered_weights[name]
                    tf_weight = tf_variables[tf_counter]

                    pre_train_ops.append(tf_weight.assign(np.transpose(weight_var, (2, 3, 1, 0))))
                    tf_counter += 1

            elif "running_" in name:  #bn mean, var
                pt_assign = reordered_weights[name]
                tf_assign = bn_variables[tf_bn_counter]

                pre_train_ops.append(tf_assign.assign(pt_assign))
                tf_bn_counter += 1

            else: #bn gamma, beta
                pt_assign = reordered_weights[name]
                tf_assign = tf_variables[tf_counter]

                pre_train_ops.append(tf_assign.assign(pt_assign))
                tf_counter += 1

    else:  #GroupNorm
        conv_variables = [v for v in tf_variables if "conv" in v.name]
        #gamma_variables = [v for v in tf_variables if "gamma" in v.name]
        #beta_variables = [v for v in tf_variables if "beta" in v.name]

        tf_conv_counter = 0
        tf_gamma_counter = 0
        tf_beta_counter = 0

        for name in weight_names:
            if "fc" in name:
                continue

            elif len(reordered_weights[name].shape) == 4:  #conv
                weight_var = reordered_weights[name]
                tf_weight = conv_variables[tf_conv_counter]

                pre_train_ops.append(tf_weight.assign(np.transpose(weight_var, (2, 3, 1, 0))))
                tf_conv_counter += 1

    return tf.group(*pre_train_ops, name='load_resnet_pretrain')


def _single_tower(net, tower_indx, input_feature, learning_rate=None, name='train'):
    _mode = mode if name is 'train' else learn.ModeKeys.INFER

    with tf.device('/gpu:%d' % tower_indx):
        with tf.variable_scope('{}_tower_{}'.format(name, tower_indx)) as scope:
            #optimizer = tf.train.AdamOptimizer(learning_rate)

            logits = net.get_logits(input_feature.image, _mode)

            loc_loss, cls_loss, tvars, extra_update_ops = net.get_loss(logits, [input_feature.loc, input_feature.cls])

            # Freeze Batch Normalization
            if FLAGS.bn_freeze:
                tvars = [t for t in tvars if "batch_normalization" not in t.name]

            #tf.get_variable_scope().reuse_variables()
            total_loss = loc_loss + cls_loss

            # Add weight decay to the loss.
            l2_loss = FLAGS.weight_decay * tf.add_n(
                [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tvars]) # if loss_filter_fn(v.name)])
            total_loss += l2_loss

            if name is 'train':
                #optimizer = tf.train.AdamOptimizer(learning_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
                grads = optimizer.compute_gradients(total_loss, tvars, colocate_gradients_with_ops=True)
            else:
                optimizer, grads = None, None
                #tf.summary.image("input_image", input_feature.image)

            #if FLAGS.verbose:
            #    for var in tf.trainable_variables():
            #        tf.summary.histogram(var.op.name, var)

            # Detection output visualize
            if name is 'valid' and FLAGS.test in ["v1", "v2"]:
                summary_images = []
                for i in range(3):
                    rect_output, _, _, _ = net.decode(logits[0][i], logits[1][i])

                    rect_output['boxes'] /= FLAGS.input_size
                    rect_output['boxes'] = tf.clip_by_value(rect_output['boxes'], 0.0, 1.0)

                    pred_img = tf.image.draw_bounding_boxes(tf.expand_dims(input_feature.image[i], 0), 
                                                            tf.expand_dims(rect_output['boxes'], 0))
                    summary_images.append(pred_img[0])

                summary_images = tf.stack(summary_images)
                tf.summary.image("pred_img", summary_images)

    return TowerResult(tvars, loc_loss, cls_loss, grads, extra_update_ops, optimizer)


def main(argv=None):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    num_gpus = len(available_gpus)
    print("num_gpus : ", num_gpus, available_gpus)
    
    with tf.Graph().as_default():

        # Get Network class and Optimizer
        global_step = tf.train.get_or_create_global_step()

        # Learning rate decay
        if "SynthText" in FLAGS.train_path:
            boundaries = [40000, 60000]
        else:
            boundaries = [4000, 8000]

        values = [FLAGS.learning_rate / pow(10, i) for i in range(3)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar('learning_rate', learning_rate)

        Hmean = tf.Variable(0.0, trainable=False, name='hmean')
        tf.summary.scalar("Hmean", Hmean)

        optimizers = []
        net = RetinaNet(FLAGS.backbone)

        # Multi gpu training code (Define graph)
        tower_grads = []
        tower_extra_update_ops = []
        #tower_train_errs = []
        tower_loc_losses = []
        tower_cls_losses = []
        input_features = net.get_input(is_train=True,
                                       num_gpus=num_gpus)

        for gpu_indx in range(num_gpus):
            tower_output = _single_tower(net, gpu_indx, input_features[gpu_indx], learning_rate)
            tower_grads.append([x for x in tower_output.grads if x[0] is not None])
            tower_extra_update_ops.append(tower_output.extra_update_ops)
            #tower_train_errs.append(tower_output.error)
            tower_loc_losses.append(tower_output.loc_loss)
            tower_cls_losses.append(tower_output.cls_loss)
            optimizers.append(tower_output.optimizer)

        if FLAGS.use_validation:
            valid_input_feature = net.get_input(is_train=False, num_gpus=1)

            # single gpu validation
            valid_tower_output = _single_tower(net, FLAGS.valid_device, valid_input_feature[0],
                                               name='valid')
            tf.summary.scalar("valid_loc_losses", valid_tower_output.loc_loss)
            tf.summary.scalar("valid_cls_losses", valid_tower_output.cls_loss)


        # Merge results
        loc_losses = tf.reduce_mean(tower_loc_losses)
        cls_losses = tf.reduce_mean(tower_cls_losses)
        grads = allreduce_grads(tower_grads)
        train_ops = []

        tf.summary.scalar("train_loc_losses", loc_losses)
        tf.summary.scalar("train_cls_losses", cls_losses)
        tf.summary.image("train_img", input_features[0].image)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_ops.append(variables_averages_op)

        # Apply the gradients
        for idx, grad_and_vars in enumerate(grads):
            with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
                # apply_gradients may create variables. Make them LOCAL_VARIABLES
                from tensorpack.graph_builder.utils import override_to_local_variable
                with override_to_local_variable(enable=idx > 0):
                    train_ops.append(optimizers[idx].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx),
                                                                 global_step=(global_step if idx==0 else None)))

        with tf.control_dependencies(tower_extra_update_ops[-1]):
            train_op = tf.group(*train_ops, name='train_op')

        # Summary
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge([s for s in summaries if 'valid_' not in s.name])

        if FLAGS.use_validation:
            valid_summary_op = tf.summary.merge([s for s in summaries if 'valid_' in s.name])
            valid_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.output,
                                                                      FLAGS.valid_dataset))
        '''
        # Print network structure
        if not os.path.exists(FLAGS.output):
            os.makedirs(os.path.join(FLAGS.output,'best_models'), exist_ok=True)
        param_stats = tf.profiler.profile(tf.get_default_graph())
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        train_info = open(os.path.join(FLAGS.output, 'train_info.txt'),'w')
        train_info.write('total_params: %d\n' % param_stats.total_parameters)
        train_info.write(str(FLAGS.flag_values_dict()))
        train_info.close()
        '''
        # Print configuration
        pprint(FLAGS.flag_values_dict())

        
        # Define config, init_op, scaffold
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        pretrain_op = load_pytorch_weight(FLAGS.use_bn, net.use_se_block)
        sync_op = _get_post_init_ops()

        # only save global variables
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        scaffold = tf.train.Scaffold(saver=saver,
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     init_fn=_get_init_pretrained())
        valid_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        best_valid_loss = 1e9
        best_valid_acc = -1

        # Define several hooks
        hooks = []
        if FLAGS.use_profile:
            profiler_hook = tf.train.ProfilerHook(save_steps=FLAGS.valid_steps,
                                                  output_dir=FLAGS.output)
            hooks.append(profiler_hook)

        if FLAGS.use_debug:
            from tensorflow.python import debug as tf_debug
            # CLI Debugger
#            cli_debug_hook = tf_debug.LocalCLIDebugHook()
#            hooks.append(cli_debug_hook)

            # Tensorboard Debugger
            tfb_debug_hook = tf_debug.TensorBoardDebugHook("127.0.0.1:9900")
            #tfb_debug_hook = tf_debug.TensorBoardDebugHook("a476cc765f91:6007")
            hooks.append(tfb_debug_hook)
        hooks = None if len(hooks)==0 else hooks

        reset_global_step = tf.assign(global_step, 0)
        
        pEval = None

        print("---------- session start")
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.output,
                                               scaffold=scaffold,
                                               hooks=hooks,
                                               config=session_config,
                                               save_checkpoint_steps=FLAGS.valid_steps,
                                               save_checkpoint_secs=None,
                                               save_summaries_steps=FLAGS.summary_steps,
                                               save_summaries_secs=None,) as sess:
            print("---------- open MonitoredTrainingSession")
            
            if "ICDAR2015" in FLAGS.train_path:
                sess.run(reset_global_step)
                
            _step = sess.run(global_step)

            if "SynthText" in FLAGS.train_path:
                print("---------- run pretrain op")
                sess.run(pretrain_op)

            print("---------- run sync op")
            sess.run(sync_op)
            
            print("---------- start training, step=", _step)

            while _step < FLAGS.max_num_steps:
                if sess.should_stop():
                    print("Done! ", _step)
                    break

                # Training
                [step_loc_loss, step_cls_loss,_ ,_step] = sess.run(
                    [loc_losses, cls_losses, train_op, global_step])

                print('STEP : %d\tTRAIN_TOTAL_LOSS : %.8f\tTRAIN_LOC_LOSS : %.8f\tTRAIN_CLS_LOSS : %.5f'
                      % (_step, step_loc_loss + step_cls_loss, step_loc_loss, step_cls_loss), end='\r')

                if _step % 50 == 0:
                    print('STEP : %d\tTRAIN_TOTAL_LOSS : %.8f\tTRAIN_LOC_LOSS : %.8f\tTRAIN_CLS_LOSS : %.5f'
                          % (_step, step_loc_loss + step_cls_loss, step_loc_loss, step_cls_loss))

                # Periodic synchronization
                if _step % 1000 == 0:
                    sess.run(sync_op)
                    
                # Validation Err
                if FLAGS.use_validation:
                    [valid_step_loc_loss, valid_step_cls_loss,  valid_summary] = sess.run([valid_tower_output.loc_loss, 
                                                                                           valid_tower_output.cls_loss, 
                                                                                           valid_summary_op])
                    if valid_summary_writer is not None: 
                        valid_summary_writer.add_summary(valid_summary, _step)

                    print('STEP : %d\tVALID_TOTAL_LOSS : %.8f\tVALID_LOC_LOSS : %.8f\tVALID_CLS_LOSS : %.5f' 
                          % (_step, valid_step_loss, valid_step_loc_loss, valid_step_cls_loss))
                    print('='*70)
                    
                # Evaluation on ICDAR2015
                if FLAGS.use_evaluation and _step % FLAGS.valid_steps == 0:
                    if "ICDAR2015" in FLAGS.train_path:
                        # reset global step -> scaffold auto save is not working!
                        saver.save(_get_session(sess), os.path.join(FLAGS.output,'model.ckpt'), global_step=_step)
                    
                    try:
                        if pEval is None:
                            print("Evaluation started at iteration {} on IC15...".format(_step))
                            eval_cmd = "CUDA_VISIBLE_DEVICES=" + str(FLAGS.valid_device) + \
                                            " python test.py" + \
                                            " --tune_from=" + os.path.join(FLAGS.output, 'model.ckpt-') + str(_step) + \
                                            " --input_size=1024" + \
                                            " --output_zip=result_" + FLAGS.test + \
                                            " --test=" + FLAGS.test + \
                                            " --nms_thresh=0.25"

                            print(eval_cmd)
                            pEval = Popen(eval_cmd, shell=True, stdout=PIPE, stderr=PIPE)

                        elif pEval.poll() is not None:
                            (scorestring, stderrdata) = pEval.communicate()

                            hmean = float(str(scorestring).strip().split(":")[3].split(",")[0].split("}")[0].strip())

                            if hmean > best_valid_acc:
                                best_valid_acc = hmean
                                best_model_dir = os.path.join(FLAGS.output, 'best_models')
                                valid_saver.save(_get_session(sess), os.path.join(best_model_dir,'model_%.2f' % (hmean*100)), global_step=_step)

                            print("test_hmean for {}-th iter : {:.4f}".format(_step, hmean))
                            sess.run(tf.assign(Hmean, hmean))
                            
                            if pEval is not None:
                                pEval.kill()
                            pEval = None

                    except Exception as e:
                        print("exception happened in evaluation ", e)
                        if pEval is not None:
                            pEval.kill()
                        pEval = None
     
if __name__ == '__main__':
    tf.app.run()
