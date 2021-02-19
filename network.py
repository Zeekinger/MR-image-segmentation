from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
from unet_attention import unet

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Net(object):
    def __init__(self, conf):
        logging.info('initialize network...')
        tf.reset_default_graph()
        self.conf = conf

        if conf.model_mode == '2d':
            self.images = tf.placeholder(tf.float32, shape=[None, None, None, self.conf.interval], name='images')
            self.labels = tf.placeholder(tf.uint8, shape=[None, None, None], name='labels')
        elif conf.model_mode == '3d':
            self.images = tf.placeholder(tf.float32, shape=[None, self.conf.interval, None, None, 1], name='images')
            self.labels = tf.placeholder(tf.uint8, shape=[None, self.conf.interval, None, None], name='labels')
        else:
            raise('model_mode '+self.conf.model_mode+' should be \'2d\' or \'3d\'')

        self.one_hot_labels = tf.one_hot(self.labels, depth=self.conf.class_num, axis=-1, name='one_hot_labels')
        self.learning_rate = tf.Variable(self.conf.learning_rate)

        logging.info('get logits...')
        self.logits = self.build_network(self.images, self.conf.class_num)
        self.loss = self.get_loss(self.logits, self.one_hot_labels)
        # self.cross_entropy = tl.cost.cross_entropy(tf.reshape(self.logits, [-1, self.conf.class_num]),
        #                                            tf.reshape(tf.cast(self.labels, tf.int32), [-1]), name='cross_entropy')
        self.cross_entropy = tf.reduce_mean(self.cross_entropy(tf.reshape(self.one_hot_labels, [-1, self.conf.class_num]),
                                                          tf.reshape(self.logits+1e-10, [-1, self.conf.class_num])))
        self.accuracy = self.get_accuracy(self.logits, self.labels)
        self.prediction = self.get_prediction(self.logits)

        self.get_summary()

    def cross_entropy(self, y_, output_map):
        pred = tf.nn.softmax(output_map)
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(pred, 1e-5, 1.0)), name="cross_entropy")

    def to_gray_image(self, image):
        image = tf.cast(image, tf.float32)
        image = image - tf.reduce_min(image)
        image = image * 255 / tf.reduce_max(image)
        return image

    def build_network(self, x, class_num):
        return unet(x, x.get_shape().as_list()[-1], class_num)

    def get_loss(self, logits, labels):
        if self.conf.loss_type == 'dice_loss':
            # dice_loss
            loss = (1 - tl.cost.dice_coe(tf.nn.softmax(logits), labels))
        elif self.conf.loss_type == 'cross_entropy':
            # weighted_cross_entropy
            prob = tf.nn.softmax(logits + 1e-10)
            loss_map = tf.multiply(labels, tf.log(tf.clip_by_value(prob, 1e-10, 1.0)))
            loss = -tf.reduce_mean(tf.multiply(loss_map, self.conf.class_weights))
        else:
            raise('unknown loss: '+self.conf.loss_type)

        return loss

    def get_prediction(self, logits):
        return tf.nn.softmax(logits)

    def get_accuracy(self, logits, labels):
        correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), -1), tf.cast(labels, tf.int64))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_summary(self):
        if self.conf.model_mode == '2d':
            for i in range(self.images.get_shape().as_list()[-1]):
                tf.summary.image('input_' + str(i), self.to_gray_image(tf.slice(self.images, (0, 0, 0, i), (1, -1, -1, 1))))
            tf.summary.image('label',
                             self.to_gray_image(tf.expand_dims(tf.slice(self.labels, (0, 0, 0), (1, -1, -1)), axis=3)))
            for i in range(1, self.prediction.get_shape().as_list()[-1]):
                tf.summary.image('output_prob_' + str(i),
                                 self.to_gray_image(tf.slice(self.prediction, (0, 0, 0, i), (1, -1, -1, 1))))
            tf.summary.image('output', self.to_gray_image(
                tf.expand_dims(tf.slice(tf.argmax(self.prediction, -1), (0, 0, 0), (1, -1, -1)), axis=3)))
        elif self.conf.model_mode == '3d':
            for i in range(self.images.get_shape().as_list()[-1]):
                tf.summary.image('input_' + str(i), self.to_gray_image(
                    tf.squeeze(tf.slice(self.images, (0, 0, 0, 0, i), (1, 1, -1, -1, 1)), axis=0)))
            tf.summary.image('label_', self.to_gray_image(
                tf.expand_dims(tf.squeeze(tf.slice(self.labels, (0, 0, 0, 0), (1, 1, -1, -1)), axis=0), axis=3)))
            for i in range(1, self.prediction.get_shape().as_list()[-1]):
                tf.summary.image('output_prob_' + str(i), self.to_gray_image(
                    tf.squeeze(tf.slice(self.prediction, (0, 0, 0, 0, i), (1, 1, -1, -1, 1)), axis=0)))
            tf.summary.image('output', self.to_gray_image(tf.expand_dims(
                tf.squeeze(tf.slice(tf.argmax(self.prediction, -1), (0, 0, 0, 0), (1, 1, -1, -1)), axis=0), axis=3)))
        else:
            raise('model_mode '+self.conf.model_mode+' should be \'2d\' or \'3d\'')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
