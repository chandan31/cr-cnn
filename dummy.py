import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.Variable(initial_value=[[0.2, 1.30, 5.5, 8.5],[9, 10.5, 11.2, 12.5]])
    b = tf.scatter_update(a, [0, 1], [[1, 0, 0, 0], [1, 0, 0, 0]])

with tf.Session(graph=g) as sess:
   sess.run(tf.global_variables_initializer())
   print sess.run(a)
   print sess.run(b)



s_theta_y = tf.gather(tf.reshape(s_theta, [-1]), y_true_index)
s_theta_c_temp = tf.reshape(tf.gather(tf.reshape(s_theta, [-1]), y_neg_index), [-1, classes_size])
s_theta_c = tf.reduce_max(s_theta_c_temp, reduction_indices=[1])
 