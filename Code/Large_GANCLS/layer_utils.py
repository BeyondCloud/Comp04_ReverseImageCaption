import tensorflow as tf

def DenseLayer(inputs, n_units, name, act=None, b_init=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    return tf.layers.dense(inputs=inputs, units=n_units, activation=act, kernel_initializer=w_init, use_bias=b_init, name=name)


def ConcatLayer(inputs, concat_dim, name):
    return tf.concat(values=inputs, axis=concat_dim, name=name)

def BatchNormLayer(inputs, is_training, name, act=None):
    output = tf.layers.batch_normalization(
                    inputs,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=is_training,
                    name=name
                )

    if act is not None:
        output = act(output)

    return output

# def reshape(inputs, shape, name):
#     return tf.reshape(inputs, shape, name)

def Conv2d(input,c_o,  kernel, stride, name, act=None, padding='VALID', b_init=False):
    k_h,k_w = kernel
    s_h,s_w = stride
    c_i = input.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(input, kernel)

        if b_init:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if act is not None:
            output = act(output, name=scope.name)

        return output

def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)

def flatten(input, name):
    input_shape = input.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        input = tf.reshape(input, [-1, dim])
    
    return input