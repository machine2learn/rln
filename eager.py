import tensorflow as tf

tf.enable_eager_execution()




x = [[2.]]

m = tf.matmul(x, x)

print(f"{m}")

# # A toy dataset of points around 3 * x + 2
# NUM_EXAMPLES = 1000
# training_inputs = tf.random_normal([NUM_EXAMPLES])
# noise = tf.random_normal([NUM_EXAMPLES])
# training_outputs = training_inputs * 3 + 2 + noise
#
# def prediction(input, weight, bias):
#   return input * weight + bias
#
# # A loss function using mean-squared error
# def loss(weights, biases):
#   error = prediction(training_inputs, weights, biases) - training_outputs
#   return tf.reduce_mean(tf.square(error))
#
# # Return the derivative of loss with respect to weight and bias
# def grad(weights, biases):
#   with tf.GradientTape() as tape:
#     loss_value = loss(weights, biases)
#   return tape.gradient(loss_value, [weights, biases])
#
# train_steps = 200
# learning_rate = 0.01
# # Start with arbitrary values for W and B on the same batch of data
# W = tf.Variable(5.)
# B = tf.Variable(10.)
#
# print("Initial loss: {:.3f}".format(loss(W, B)))
#
# for i in range(train_steps):
#   dW, dB = grad(W, B)
#   W.assign_sub(dW * learning_rate)
#   B.assign_sub(dB * learning_rate)
#   if i % 20 == 0:
#     print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))
#
# print("Final loss: {:.3f}".format(loss(W, B)))
# print("W = {}, B = {}".format(W.numpy(), B.numpy()))

x = tf.Variable(10.)

checkpoint = tf.train.Checkpoint(kashk=x)  # save as "x"

x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(save_path)

print(f'amir-{x}')  # => 2.0
