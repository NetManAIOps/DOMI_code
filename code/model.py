# -*- coding: utf-8 -*-
import tfsnippet as spt
import functools
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from config import ExpConfig
config = ExpConfig()


@spt.global_reuse
@add_arg_scope
def q_net(x, observed=None, n_z=None, is_training=False, is_initializing=False):
    """
    Inference net
    param x: input X, multivariate time series data.
    return q net structure.
    """
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )
    print("="*10+"qnet"+"="*10)

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size2,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.elu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        print("qx:%s"%x.get_shape())
        h_x = tf.reshape(
            tf.to_float(x),
            [-1, config.timeLength, config.metricNumber, 1]
            if config.channels_last 
            else [-1, 1, config.timeLength, config.metricNumber]
            )
        print("q1:%s"%h_x.get_shape())
        h_x = spt.layers.resnet_conv2d_block(
            h_x, 1, kernel_size=(config.kernel_size1, 1), strides=(config.strides1, 1)
        )
        print("q2:%s"%h_x.get_shape())
        h_x = spt.layers.resnet_conv2d_block(
            h_x, 1, kernel_size=(config.kernel_size1, 1), strides=(config.strides1, 1)
        )
        print("q3:%s"%h_x.get_shape())
        h_x = spt.layers.resnet_conv2d_block(
            h_x, 1, kernel_size=(config.kernel_size2, 1), strides=(config.strides2, 1)
        )
        print("q4:%s"%h_x.get_shape())
        h_x = spt.layers.resnet_conv2d_block(
            h_x, 1, kernel_size=(config.kernel_size2, 1), strides=(config.strides2, 1)
        )
        print("q5:%s"%h_x.get_shape())

    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    print("q6:%s" % h_x.get_shape())

    # sample y ~ q(y|x)
    c_logits = spt.layers.dense(h_x, config.n_c, name='c_logits')
    c = net.add('c', spt.Categorical(c_logits))
    c_one_hot = tf.one_hot(c, config.n_c, dtype=tf.float32)
    print("qc:%s, %s, %s" % (c_logits.shape, c.shape, c_one_hot.shape))
    h_z = h_x

    # sample z ~ q(z|x)
    z_mean = spt.layers.dense(h_z, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_z, config.z_dim, name='z_logstd', activation_fn=tf.nn.elu) + config.std_epsilon
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z, group_ndims=1)
    print("q7:%s, %s, %s" % (z_mean.get_shape(), z_logstd.get_shape(), z.get_shape()))

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, is_training=False, is_initializing=False):
    """
    Generative net
    return p net structure.
    """
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    def make_component(i):
        normal = spt.Normal(
            mean=tf.get_variable('mean_{}'.format(i), shape=[1, config.z_dim],
                                 dtype=tf.float32, trainable=True),
            logstd=tf.maximum(
                tf.get_variable('logstd_{}'.format(i), shape=[1, config.z_dim],
                                dtype=tf.float32, trainable=True),
                -1.
            )
        )
        return normal.expand_value_ndims(1)

    components = [make_component(i) for i in range(config.n_c)]
    mixture = spt.Mixture(
        categorical=spt.Categorical(logits=tf.zeros([1, config.n_c])),
        components=components,
        is_reparameterized=True
    )
    z = net.add('z', mixture, n_samples=n_z)

    print("="*10+"pnet"+"="*10)
    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size2,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.elu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        print("px:%s"%z.get_shape())
        h_z = spt.layers.dense(
            z, int(config.timeLength / (config.strides1 ** 2) / (config.strides2 ** 2) * int(config.metricNumber))
        )
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(
                int(config.timeLength / (config.strides1 ** 2) / (config.strides2 ** 2)),
                int(config.metricNumber), 1
            )
            if config.channels_last else (
                1, int(config.timeLength / (config.strides1 ** 2) / (config.strides2 ** 2)),
                int(config.metricNumber)
            )
        )
        print("p1:%s"%h_z.get_shape())
        h_z = spt.layers.resnet_deconv2d_block(
            h_z, 1, kernel_size=(config.kernel_size2, 1), strides=(config.strides2, 1)
        )
        print("p2:%s"%h_z.get_shape())
        h_z = spt.layers.resnet_deconv2d_block(
            h_z, 1, kernel_size=(config.kernel_size2, 1), strides=(config.strides2, 1)
        )
        print("p3:%s"%h_z.get_shape())
        h_z = spt.layers.resnet_deconv2d_block(
            h_z, 1, kernel_size=(config.kernel_size1, 1), strides=(config.strides1, 1)
        )
        print("p4:%s"%h_z.get_shape())
        h_z = spt.layers.resnet_deconv2d_block(
            h_z, 1, kernel_size=(config.kernel_size1, 1), strides=(config.strides1, 1)
        )
        print("p5:%s"%h_z.get_shape())

    # sample x ~ p(x|z)
    x_mean = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', name='x_mean',
        channels_last=config.channels_last
    )
    x_logstd = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', name='x_logstd',
        channels_last=config.channels_last, activation_fn=tf.nn.elu, 
    ) + config.std_epsilon
    x = net.add('x', spt.Normal(mean=x_mean, logstd=x_logstd), n_samples=n_z, group_ndims=3)
    print("p6:%s, %s, %s" % (x_mean.get_shape(), x_logstd.get_shape(), x.get_shape()))

    return net
