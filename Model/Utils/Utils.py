import tensorflow as tf

def pad_tensor(ys):
    length = [y.shape[0] for y in ys]
    max_length = max(length)
    ys = tf.stack([tf.pad(y, tf.constant([[0, max_length - y.shape[0]], [0, 0]])) for y in ys])
    mask = tf.tile(tf.reshape(tf.range(0, max_length, dtype=tf.int32), (1, -1)), (len(length), 1))
    mask = mask < tf.reshape(tf.constant(length), (-1, 1))
    return ys, mask

def test_pad_tensor():
    ys = [tf.random.uniform((10, 5), minval=0, maxval=100, dtype=tf.int32),
          tf.random.uniform((20, 5), minval=0, maxval=100, dtype=tf.int32),
          tf.random.uniform((5, 5), minval=0, maxval=100, dtype=tf.int32)]
    ys, mask = pad_tensor(ys)
    print(ys)
    print(mask)
    print(ys.shape)
    print(mask.shape)

if __name__ == '__main__':
    test_pad_tensor()