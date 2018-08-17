from __future__ import division
from __future__ import absolute_import
import tensorflow as tf


class baseline(object):
    def get_baseline_value(self):
        pass

    def update(self, target):
        pass


class ReactiveBaseline(baseline):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.baseline = tf.Variable(0.0, trainable=False)

    def get_baseline_value(self):
        return self.baseline

    def update(self, target):
        self.baseline = tf.add((1 - self.learning_rate) * self.baseline,
                        self.learning_rate * target)
