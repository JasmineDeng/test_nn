import tensorflow as tf
import numpy as np
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

def normalize_rgb(data):
	count = 0
	red = np.empty(len(data))
	green = np.empty(len(data))
	blue = np.empty(len(data))
	for i in range(len(data)):
		red[i] = np.average(data[i][:1024])
		green[i] = np.average(data[i][1024:2048])
		blue[i] = np.average(data[i][2048:])
	red_avg = np.average(red)
	green_avg = np.average(green)
	blue_avg = np.average(blue)
	print('calculated avg')
	for i in range(len(data)):
		for j in range(1024):
			data[i][j] -= red_avg
			data[i][j + 1024] -= green_avg
			data[i][j + 2048] -= blue_avg
		count += 1
		if count % 1000 == 0:
			print('normalized %d images' % (count))
	print('done normalizing')

def create_var(shape):
	return tf.Variable(tf.truncated_normal(shape, 0, stddev=0.01, dtype=tf.float32), validate_shape=False)

def shuffle_data(d):
	arr = zip(d['data'], d['labels'])
	random.shuffle(arr)
	return zip(*arr)

def conv_layer(inp, weight_shape, bias_shape, stride_shape, ksize_shape):
	weights = create_var(weight_shape)
	bias = create_var(bias_shape)
	conv = tf.nn.conv2d(inp, weights, strides=[1, 1, 1, 1], padding='SAME')
	activation = activation_func(conv + bias)
	pool = tf.nn.max_pool(activation, ksize=ksize_shape, strides=stride_shape, padding='SAME')
	return pool

def fc_layer(inp, weight_shape, bias_shape, reshape):
	weights = create_var(weight_shape)
	bias = create_var(bias_shape)
	new_inp = tf.reshape(inp, reshape)
	activation = activation_func(tf.matmul(new_inp, weights) + bias)
	return activation

def dropout(inp, keep_prob):
	return tf.nn.dropout(inp, keep_prob)

def run_net(inp_dict, train_batch):
	count = 0
	for file_name in train_batch:
		data, labels = shuffle_data(unpickle(file_name))
		normalize_rgb(data)
		labels = reshape_labels(labels)
		for k in range(batch_size):
			data_batch = data[k:k + batch_size]
			labels_batch = labels[k:k + batch_size]
			train_step.run(feed_dict=inp_dict.update({ x: data_batch, y_actual: labels_batch, keep_prob: k_p }))
			if k == 99:
				train_accuracy = accuracy.eval(feed_dict={ x: data_batch, y_actual: labels_batch, keep_prob: k_p})
				print("step %d, training accuracy %g"%(count, train_accuracy))
		count += 1

def validation_test(data_name, inp_dict):
	test_data = unpickle(data_name)
	avg = 0
	for i in range(100):
		labels = reshape_labels(test_data['labels'])[i:i + 100]
		test_accuracy = accuracy.eval(feed_dict=inp_dict.update({ x: test_data['data'][i:i+100], y_actual: labels, keep_prob: k_p }))
		avg += test_accuracy
		if i % 10 == 0:
			print("test accuracy for batch %d is %g"%(i, test_accuracy))
	print("accuracy overall is " + str(avg / 100.0))
	return avg / 100.0

# declare placeholder variables
x = tf.placeholder(tf.float32, [100, 3072])
keep_prob = tf.placeholder(tf.float32)
y_actual = tf.placeholder(tf.float32, [100, 10])

feat_1_temp = tf.placeholder(tf.int32)
feat_2_temp = tf.placeholder(tf.int32)
fc_neuron_temp = tf.placeholder(tf.int32)
feat_1 = tf.Variable((0))
feat_2 = tf.Variable((0))
fc_neuron = tf.Variable((0))
feat_1.assign(feat_1_temp)
feat_2.assign(feat_2_temp)
fc_neuron.assign(fc_neuron_temp)

# reshape into an image with 3 color channels
x_image = tf.reshape(x, [100, 32, 32, 3])

# so can easily change variables for entire network
activation_func = tf.nn.relu
num_conv = 2
conv_shape = ((32 // (num_conv ** 2)) ** 2) * feat_2
batch_size = 100
stride_k_size = [1, 2, 2, 1]
k_p = 0.5

# convolutional layer 1
conv_layer_1 = conv_layer(x_image, [5, 5, 3, feat_1], [feat_1], stride_k_size, stride_k_size)
#convolutional layer 2
conv_layer_2 = conv_layer(conv_layer_1, [5, 5, feat_1, feat_2], [feat_2], stride_k_size, stride_k_size)

# fully connected layer
fc_layer_1 = fc_layer(conv_layer_2, [conv_shape, fc_neuron], [fc_neuron], [batch_size, conv_shape])
dropout(fc_layer_1, keep_prob)

# reshape into 10 for later softmax
y_result = tf.matmul(fc_layer_1, create_var([fc_neuron, 10])) + create_var([10])

# construct rest of graph
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_result))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_result, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

feat_1_inp = [2, 4, 8, 16, 32, 64]
feat_2_inp = [2, 4, 8, 16, 32, 64]
fc_neuron_inp = [32, 64, 128, 256, 512, 1024]

max_elem = ({}, float('-inf'))
train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
for f1 in feat_1_inp:
	for f2 in feat_2_inp:
		for fc in fc_neuron_inp:
			sess = tf.InteractiveSession()
			sess.run(tf.initialize_all_variables())
			inp_dict = {feat_1: f1, feat_2: f2, fc_neuron: fc}
			run_net(inp_dict, train_batch)
			max_elem = max((inp_dict, validate_test('data_batch_5', inp_dict)), max_elem, key=lambda x:x[0])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
train_batch.append('data_batch_5')
run_net(max_elem[0], train_batch)
validate_test('test_batch')
