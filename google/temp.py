import tensorflow as tf

'''声明的变量需要初始化，初始化方式有下面两种：初始化变量方式1'''
# a = tf.Variable(1.0)
# b = tf.constant(4.0)
# c = tf.add(a, b)
# # way one
# init_op = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init_op)
# res = sess.run(c)
# print(res)
# sess.close()

'''初始化变量方式2'''
# a = tf.Variable(1.0)
# b = tf.constant(4.0)
# c = tf.add(a, b)
# way two
# sess = tf.InteractiveSession()
# a.initializer.run()
# res = c.eval()
# print(res)


'''变量运算自增操作'''
# state = tf.Variable(0, name='counter')
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)  #初始化操作
#     print(sess.run(state))
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))


'''获取计算后的结果'''
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.multiply(input1, intermed)
# with tf.Session() as sess:
#     result = sess.run([mul, intermed])
#     print(result)

'''
占位符，占位符不需要初始化，
需要在运行的时候指定对应的值，
而变量在运算之前需要进行初始化操作
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.0], input2: [2.]}))
