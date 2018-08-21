import os
import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("use_gpu", False, "True if we want to use the gpu.")

tf.flags.DEFINE_integer("min_inner_size", 2, "The lower inner size inclusive.")
tf.flags.DEFINE_integer("max_inner_size", 2 ** 10,
                        "The upper inner size inclusive.")
tf.flags.DEFINE_integer("outer_size", 2 ** 4, "The outer size.")

tf.flags.DEFINE_integer("batch_size", 2 ** 6, "The final vector size.")
tf.flags.DEFINE_integer("repeats", 2 ** 6, "The final vector size.")


def build_multiply_graph(
        batch_size,
        inner_size,
        outer_size,
        b_placeholder=True):
    """

    Args:
        batch_size:
        inner_size:
        outer_size:
        b_placeholder:

    Returns:

    """
    a_shape = batch_size, inner_size
    b_shape = inner_size, outer_size

    g = tf.Graph()

    with g.as_default():
        a = tf.placeholder(dtype=tf.float32, shape=a_shape, name="a")

        if b_placeholder:
            b = tf.placeholder(dtype=tf.float32, shape=b_shape, name="b")
        else:
            b = tf.get_variable(name="b", shape=b_shape, dtype=tf.float32)
            b_replace = tf.placeholder(dtype=tf.float32, shape=b_shape,
                                       name="b_replace")
            b_assign = tf.assign(ref=b, value=b_replace)

        c = tf.matmul(a, b)

    if b_placeholder:
        return g, (a, b)

    return g, (a, b, b_replace, b_assign)
