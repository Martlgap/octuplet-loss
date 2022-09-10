import tensorflow as tf


def reduce_res(batch, res: int):
    """Down- & Up-sampling of images in a batch

    :param batch: batch tensor shape=[batch_size, height, width, channels]
    :param res: the desired resolution as integer
    :return: batch
    """

    def reduce_res_fn(inputs):
        img_hr = inputs[0]
        dim_lr = inputs[1]
        dim_hr = tf.shape(img_hr)[0]
        img_rlr = tf.image.resize(img_hr, (dim_lr, dim_lr), antialias=True, method=tf.image.ResizeMethod.BICUBIC)
        img_lr = tf.image.resize(img_rlr, (dim_hr, dim_hr), antialias=False, method=tf.image.ResizeMethod.BICUBIC)
        return tf.saturate_cast(img_lr, dtype=img_hr.dtype)

    batch_size = tf.shape(batch)[0]
    batch_res = tf.repeat(
        tf.gather([res], tf.random.uniform([], minval=0, maxval=1, dtype=tf.int32)),
        repeats=batch_size,
    )
    return tf.map_fn(
        reduce_res_fn, elems=(batch, batch_res), fn_output_signature=batch.dtype
    )  # Apply function to each element of batch
