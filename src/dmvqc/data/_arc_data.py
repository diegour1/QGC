

import numpy as np
from  sklearn import model_selection

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import scipy

def _load_arc():
    dataset_size = 12_000

    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(dataset_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(dataset_size, dtype=tf.float32))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)

    X_densities = x2_dist.prob(x_samples[:,1]) * x1.prob(x_samples[:,0])

    # True densities
    x2 = tfp.distributions.Normal(loc = 0., scale = 4.)
    x1 = tfp.distributions.Normal(loc = .25 * tf.square(x_samples[:,1]), scale = tf.ones(12_000, dtype=tf.float32))

    # Mesh
    x, y = np.mgrid[-10:40:(50/120), -15:15:(30/120)]
    pos = np.dstack((x, y))
    X_plot = pos.reshape([14400,2])

    real_prob = x2_dist.prob(X_plot[:,1]) * scipy.stats.norm(0.25 * np.square(X_plot[:,1]), 1).pdf(X_plot[:,0])

    batch_size = 32
    X_train, X_test = model_selection.train_test_split(x_samples.numpy(), test_size=1/6)

    train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    batched_train_data = train_dataset.batch(batch_size)