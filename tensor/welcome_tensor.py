import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Testing")

print(sess.run(hello))

