import tensorflow as tf
import random

def unpickle(path):
    import cPickle
    file = 'cifar-10-batches-py/' + str(path)
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def reshape_labels(arr):
	new_arr = []
	for elem in arr:
		temp = [float(0) for i in range(0, 10)]
		temp[elem] = float(1)
		new_arr.append(temp)
	return new_arr

def create_var(shape):
	return tf.Variable(tf.truncated_normal(shape, 0, stddev=0.01, dtype=tf.float32))

def shuffle_data(d):
	arr = zip(d['data'], d['labels'])
	random.shuffle(arr)
	return zip(*arr)

x = tf.placeholder(tf.float32, [100, 3072])
y_actual = tf.placeholder(tf.float32, [100, 10])
x_image = tf.reshape(x, [100, 32, 32, 3])

weights_1 = create_var([5, 5, 3, 32])
bias_1 = create_var([16])

# so can easily change activation function for entire network
activation_func = tf.nn.relu

# convolutional layer with sigmoid and max pooling
# computes 4 features
conv_layer_1 = tf.nn.conv2d(x_image, weights_1, strides=[1, 1, 1, 1], padding='SAME')
activation_1 = activation_func(conv_layer_1 + bias_1)
pool_1 = tf.nn.max_pool(activation_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# second convolutional layer with sigmoid and max pooling
# computes 8 features
weights_2 = create_var([5, 5, 32, 16])
bias_2 = create_var([16])

conv_layer_2 = tf.nn.conv2d(pool_1, weights_2, strides=[1, 1, 1, 1], padding='SAME')
activation_2 = activation_func(conv_layer_2 + bias_2)
pool_2 = tf.nn.max_pool(activation_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

fc_neuron = 64

# fully connected layer
weights_3 = create_var([16 * 8 * 8, fc_neuron])
bias_3 = create_var([fc_neuron])

pool_3 = tf.reshape(pool_2, [100, 16 * 8 * 8])
activation_3 = activation_func(tf.matmul(pool_3, weights_3) + bias_3)

keep_prob = tf.placeholder(tf.float32)
dropout_1 = tf.nn.dropout(activation_3, keep_prob)

# reshape into 10 
weight_reshape = create_var([fc_neuron, 10])
bias_reshape = create_var([10])
max_pool = tf.reshape(activation_3, [100, fc_neuron])

dropout_2 = tf.nn.dropout(max_pool, keep_prob)

# y_result = tf.nn.softmax(inp)

# construct rest of graph
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y_result), reduction_indices=[1]))

y_result = tf.matmul(max_pool, weight_reshape) + bias_reshape
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_result))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_result, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
count = 1

for file_name in train_batch:
	data, labels = shuffle_data(unpickle(file_name))
	labels = reshape_labels(labels)
	for k in range(100):
		train_step.run(feed_dict={ x: data[k:k + 100], y_actual: labels[k:k + 100], keep_prob: 0.5 })
		if k == 99:
			train_accuracy = accuracy.eval(feed_dict={ x: data[k:k + 100], y_actual: labels[k:k + 100], keep_prob: 0.5})
			print("step %d, training accuracy %g"%(count, train_accuracy))
	count += 1

test_data = unpickle('test_batch')
avg = 0
for i in range(100):
	labels = reshape_labels(test_data['labels'])[i:i + 100]
	test_accuracy = accuracy.eval(feed_dict={ x: test_data['data'][i:i+100], y_actual: labels, keep_prob: 0.5 })
	avg += test_accuracy
	if i % 10 == 0:
		print("test accuracy for batch %d is %g"%(i, test_accuracy))
print("test accuracy overall is " + str(avg / 100.0))

