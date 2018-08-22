import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tf_speed_test import utils as speed_utils

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean("use_gpu", False, "True if we want to use the gpu.")

tf.flags.DEFINE_integer("min_inner_size", 2, "The lower inner size inclusive.")
tf.flags.DEFINE_integer("max_inner_size", 2 ** 13,
                        "The upper inner size inclusive.")
tf.flags.DEFINE_integer("sizes", 2 ** 4,
                        "The number of sizes to try inclusive.")
tf.flags.DEFINE_integer("outer_size", 2 ** 4, "The outer size.")

tf.flags.DEFINE_integer("batch_size", 2 ** 8, "The final vector size.")
tf.flags.DEFINE_integer("repeats", 2 ** 8, "The final vector size.")

inner_sizes = np.logspace(start=np.log2(FLAGS.min_inner_size),
                          stop=np.log2(FLAGS.max_inner_size),
                          num=FLAGS.sizes, base=2, dtype=np.int32)

times_ph = [speed_utils.time_placeholder_graph(batch_size=FLAGS.batch_size,
                                               inner_size=inner_size,
                                               outer_size=FLAGS.outer_size,
                                               repeats=FLAGS.repeats)
            for inner_size in inner_sizes]

times_ph_mean, times_ph_std = zip(*times_ph)

fig, ax = plt.subplots()
ymax = np.max(times_ph_mean)

ax.scatter(inner_sizes, times_ph_mean, label="Placeholder")
ax.set_ylim([0, ymax * 1.1])
ax.set_xlabel("Inner size of matrix product")
ax.set_ylabel(f"Mean time taken (s) over {FLAGS.repeats} repetitions")
ax.legend()
plt.show()
