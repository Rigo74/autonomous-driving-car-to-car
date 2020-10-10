import tensorflow as tf
tf_v1 = tf.compat.v1


def gpu_configuration():
    config = tf_v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf_v1.Session(config=config)
    tf_v1.keras.backend.set_session(sess)
