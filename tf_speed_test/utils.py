import time
import numpy as np
import tensorflow as tf


def build_multiply_graph(batch_size,
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
        a_ = tf.placeholder(dtype=tf.float32, shape=a_shape, name="a")

        if b_placeholder:
            b_ = tf.placeholder(dtype=tf.float32, shape=b_shape, name="b")
            c = tf.matmul(a_, b_)
            return g, (a_, b_, c)

        b = tf.get_variable(name="b", shape=b_shape, dtype=tf.float32)
        b_replace_ = tf.placeholder(dtype=tf.float32, shape=b_shape,
                                    name="b_replace")
        b_assign = tf.assign(ref=b, value=b_replace_, name="b_assign")
        c = tf.matmul(a_, b)
        return g, (a_, b, c, b_replace_, b_assign)


def time_placeholder_graph(batch_size, inner_size, outer_size, repeats):
    g, (a_, b_, c) = build_multiply_graph(batch_size=batch_size,
                                          inner_size=inner_size,
                                          outer_size=outer_size)

    b_val = np.random.uniform(size=b_.shape)
    sess = tf.Session(graph=g)

    def _time_operation():
        a_val = np.random.uniform(size=a_.shape)
        fetches = c
        feed_dict = {a_: a_val, b_: b_val}
        now = time.time()
        sess.run(fetches=fetches, feed_dict=feed_dict)
        end = time.time()
        return end - now

    results = np.array([_time_operation() for _ in range(repeats)])

    return np.mean(results), np.std(results)
