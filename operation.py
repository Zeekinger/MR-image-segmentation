#!/usr/bin/env python
# coding=utf-8

from __future__ import division

import os
import shutil
import copy
import time
import skimage.measure
import numpy as np
import nibabel as nib
import tensorflow as tf
import scipy.ndimage as ndi
from medpy import metric
from skimage.transform import resize
from scipy.ndimage.measurements import label
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Operation(object):
    def __init__(self, data_provider, conf):
        self.conf = conf
        self.data_provider = data_provider
        self.learning_rate = tf.Variable(conf.learning_rate)
        self.training_iters = len(self.data_provider.train_patch_list)//self.conf.batch_size

        logging.info('batch_size: {}, learning_rate: {}, optimizer: {}'.format(conf.batch_size, conf.learning_rate,
                                                                               conf.optimizer))
        logging.info('epochs: {}, training_iters: {}'.format(conf.epochs, self.training_iters))

    def train(self, net):
        tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope('train'):
            train_op = self.get_train_op(net)
            summary_op = tf.summary.merge_all()

        logging.info("Start optimization")
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(self.conf.log_path, graph=sess.graph)
            sess.run(tf.global_variables_initializer())

            if self.conf.need_restore:
                ckpt = tf.train.get_checkpoint_state(self.conf.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.info("Removing '{:}'".format(self.conf.log_path))
                shutil.rmtree(self.conf.log_path, ignore_errors=True)
                logging.info("Removing '{:}'".format(self.conf.model_path))
                shutil.rmtree(self.conf.model_path, ignore_errors=True)

            if not os.path.exists(self.conf.log_path):
                logging.info("Allocating '{:}'".format(self.conf.log_path))
                os.makedirs(self.conf.log_path)
            if not os.path.exists(self.conf.model_path):
                logging.info("Allocating '{:}'".format(self.conf.model_path))
                os.makedirs(self.conf.model_path)
            if not os.path.exists(self.conf.prediction_path):
                logging.info("Allocating '{:}'".format(self.conf.prediction_path))
                os.makedirs(self.conf.prediction_path)

            # self.print_global_variable_info(sess)
            for epoch in range(self.conf.epochs):
                for step in range((epoch*self.training_iters), ((epoch+1)*self.training_iters)):
                    batch_x, batch_y = self.data_provider.get_random_batch(self.conf.batch_size)
                    batch_x, batch_y = self.preprocess(batch_x, batch_y)

                    _, loss = sess.run((train_op, net.loss),
                                       feed_dict={net.images: batch_x,
                                                  net.labels: batch_y})

                    if step % (self.training_iters // self.conf.interval) == 0:
                        self.save(sess, os.path.join(self.conf.model_path, 'model_' + str(step) + '_%.4f' % loss + '.ckpt'))

                    if step % self.conf.display_step == 0:
                        self.output_minibatch_stats(sess, net, summary_op, summary_writer, epoch, step, batch_x, batch_y)

            logging.info("Optimization Finished!")

    def get_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Args:
          learning_rate: A scalar or `Tensor` learning rate.

        Returns:
          An instance of an optimizer.

        Raises:
          ValueError: if self.conf.optimizer is not recognized.
        """
        if self.conf.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=0.95,
                epsilon=1e-8)
        elif self.conf.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=0.1)
        elif self.conf.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self.conf.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=-0.5,
                initial_accumulator_value=0.1,
                l1_regularization_strength=0.0,
                l2_regularization_strength=0.0)
        elif self.conf.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=0.9,
                name='Momentum',
                use_locking=False,
                use_nesterov=True)
        elif self.conf.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=0.9,
                momentum=0.9,
                epsilon=1e-10)
        elif self.conf.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.conf.optimizer)
        return optimizer

    def get_train_op(self, net):
        train_op = self.get_optimizer(self.learning_rate).minimize(net.loss)
        return train_op

    def preprocess(self, img, mask):
        img = self.img_preprocess(img)
        mask = self.mask_preprocess(mask)
        return img, mask

    def mask_preprocess(self, mask):
        if self.conf.model_mode == '2d':
            mask = mask[..., mask.shape[-1]//2]
        elif self.conf.model_mode == '3d':
            mask = np.transpose(mask, (0, 3, 1, 2))
        else:
            raise ('model_mode ' + self.conf.model_mode + ' should be \'2d\' or \'3d\'')

        return mask

    def img_preprocess(self, img):
        if self.conf.model_mode == '2d':
            pass
        elif self.conf.model_mode == '3d':
            img = np.transpose(img, (0, 3, 1, 2))[..., np.newaxis]
        else:
            raise ('model_mode ' + self.conf.model_mode + ' should be \'2d\' or \'3d\'')

        return img

    def output_minibatch_stats(self, sess, net, summary_op, summary_writer, epoch, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, ce, acc, prediction = sess.run([summary_op,
                                                           net.loss,
                                                           net.cross_entropy,
                                                           net.accuracy,
                                                           net.prediction],
                                                          feed_dict={net.images: batch_x,
                                                                     net.labels: batch_y})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        if self.conf.model_mode == '2d':
            error_rate = 100.0 - (100.0 * np.sum(np.argmax(prediction, -1) == batch_y) / (prediction.shape[0]*prediction.shape[1]*prediction.shape[2]))
        elif self.conf.model_mode == '3d':
            error_rate = 100.0 - (100.0 * np.sum(np.argmax(prediction, -1) == batch_y) / (prediction.shape[0]*prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
        else:
            raise('model_mode '+self.conf.model_mode+' should be \'2d\' or \'3d\'')
        logging.info("Epoch {}, Iter {}/{}, Minibatch Loss= {:.4f}, Minibatch cross_entropy= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(epoch, step%self.training_iters, self.training_iters, loss, ce, acc, error_rate))

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        logging.info("Model save at file: %s" % model_path)

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['train/'])
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver()
        #print(model_path)
        saver.restore(sess, 'C:\\Users\\12638\\dfsdata2\\zhangyao_data\\DB\\SpineSagT2W\\standard_data\\trained\\model_57144_0.0128.ckpt')
        logging.info("Model restored from file: %s" % model_path)


############################################################ eval

    def single_connected_domain_filter(self, volume, label):
        output = copy.deepcopy(volume)
        label_3D = skimage.measure.label(volume == label)
        max_label = np.max(label_3D)
        max_size = 0
        max_label_id = 0
        for label in range(1, max_label + 1):
            label_size = len(np.where(label_3D == label)[0])
            if label_size > max_size:
                max_size = label_size
                max_label_id = label
        output[np.where(label_3D != max_label_id)] = 0
        if max_size < 256:
            output[np.where(label_3D == max_label_id)] = 0
        return output

    def read_sample_from_volume(self, index, volume):
        """
        read consecutive slices from a given volume centered at the index
        If the index is at the beginning of ending of the volume, this method will duplicate the starting and ending slices

        :param index: the index of the slice in a volume
        :param volume: the input volume data in numpy array format
        :return: the consecutive slices with slice range of a volume in numpy array
        """
        data_sample = []
        for i in range(index-self.conf.interval//2, index+self.conf.interval//2+1):
            data_sample.append(volume[:, :, np.clip(i, 0, volume.shape[2]-1)])
        data_sample = np.transpose(np.asarray(data_sample, dtype=np.float32), (1, 2, 0))
        return data_sample


    def eval(self, sess, net, evaluation, inference):
        logging.info('Validation Start')
        spine_score_list = []
        duration_list = []

        val_volume_list = self.data_provider.get_val_volume_list()
        for volume_filename in val_volume_list:
            start_time = time.time()
            case_id, data_volume,  affine = self.data_provider.read_volume(volume_filename, with_seg=False)

            seg_prob_volume = np.zeros(data_volume.shape + (self.conf.class_num, ), dtype=np.float32)

            # for 2d seg
            if self.conf.model_mode == '2d':
                for i in range(data_volume.shape[2]):
                    data_sample = self.read_sample_from_volume(i, data_volume)
                    #gt_seg_sample = self.read_sample_from_volume(i, gt_seg_volume)

                    #if np.max(gt_seg_sample) == 0:
                    #    continue

                    seg_prob_slice = sess.run([net.prediction], feed_dict={net.images: data_sample[np.newaxis, ...]})

                    seg_prob_slice = np.squeeze(seg_prob_slice)
                    seg_prob_volume[..., i, :] = seg_prob_slice

            # get seg_volume
            final_seg_prob_volume = seg_prob_volume

            final_seg_prob_volume = final_seg_prob_volume.astype(np.float32)
            final_seg_volume = np.argmax(final_seg_prob_volume, axis=-1).astype(np.uint8)

            duration = time.time() - start_time
            duration_list.append(duration)
            logging.info('case: {}, len: {}, duration: {}'.format(case_id, data_volume.shape[2], duration))

            # inference
            if inference:
                nii_volume = nib.Nifti1Image(final_seg_volume, affine)
                nib.save(nii_volume, os.path.join(self.conf.prediction_path, case_id.replace('Case', 'mask_case')+'.nii'))
                logging.info('case: {}, len: {}'.format(case_id, data_volume.shape[2]))

            # evaluation
            if evaluation:
                #spine_dice = metric.binary.dc(final_seg_volume==1, gt_seg_volume==1)
                #logging.info('case: {}, len: {}, spine dice: {}'.format(case_id, data_volume.shape[2], spine_dice))
                #spine_score_list.append(spine_dice)
                pass

        if evaluation:
            #logging.info(
                #'validation len: {}, spine dice per case: {}'.format(
                  #  len(spine_score_list), np.mean(spine_score_list)))
            pass
        logging.info('mean duration: {}'.format(np.mean(duration_list)))

    def eval_with_volumes(self, net, evaluation, inference):
        logging.info('Validation Start')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.conf.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.restore(sess, ckpt.model_checkpoint_path)
            self.eval(sess, net, evaluation, inference)
