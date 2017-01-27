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
	return tf.Variable(tf.truncated_normal(shape, 0, stddev=0.01, dtype=tf.float32))

def shuffle_data(d):
	arr = zip(d['data'], d['labels'])
	random.shuffle(arr)
	return zip(*arr)

def conv_layer(inp, weight_shape, bias_shape, act_func):
	weights = create_var(weight_shape)
	bias = create_var(bias_shape)
	conv = tf.nn.conv2d(inp, weights, strides=[1, 1, 1, 1], padding='SAME')
	activation = act_func(conv + bias)
	pool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return pool

def fc_layer(inp, weight_shape, bias_shape, reshape_shape, act_func):
	weights = create_var(weight_shape)
	bias = create_var(bias_shape)
	new_inp = tf.reshape(inp, reshape_shape)
	activation = act_func(tf.matmul(new_inp, weights) + bias)
	return activation

def run_validation(feat_1, feat_2, fc_neuron):
	x = tf.placeholder(tf.float32, [100, 3072])
	y_actual = tf.placeholder(tf.float32, [100, 10])
	x_image = tf.reshape(x, [100, 32, 32, 3])

	# so can easily change activation function for entire network
	activation_func = tf.nn.relu

	# convolutional layer with sigmoid and max pooling
	# computes 4 features
	conv_layer_1 = conv_layer(x_image, [5, 5, 3, feat_1], [feat_1], activation_func)

	# second convolutional layer with sigmoid and max pooling
	# computes 8 features
	conv_layer_2 = conv_layer(conv_layer_1, [5, 5, feat_1, feat_2], [feat_2], activation_func)

	conv_shape = (32 // 4) ** 2 * feat_2

	# fully connected layer
	fc_layer_1 = fc_layer(conv_layer_2, [conv_shape, fc_neuron], [fc_neuron], [100, conv_shape], activation_func)

	keep_prob = tf.placeholder(tf.float32)
	dropout_1 = tf.nn.dropout(fc_layer_1, keep_prob)

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

	train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
	count = 1

	for file_name in train_batch:
		data, labels = shuffle_data(unpickle(file_name))
		labels = reshape_labels(labels)
		for k in range(100):
			train_step.run(feed_dict={ x: data[k:k + 100], y_actual: labels[k:k + 100], keep_prob: 0.5 })
		count += 1

	test_data = unpickle('data_batch_5')
	avg = 0
	for i in range(100):
		labels = reshape_labels(test_data['labels'])[i:i + 100]
		test_accuracy = accuracy.eval(feed_dict={ x: test_data['data'][i:i+100], y_actual: labels, keep_prob: 0.5 })
		avg += test_accuracy
	return float(avg) / 100.0

def run_net(feat_1, feat_2, fc_neuron):
	x = tf.placeholder(tf.float32, [100, 3072])
	y_actual = tf.placeholder(tf.float32, [100, 10])
	x_image = tf.reshape(x, [100, 32, 32, 3])

	# so can easily change activation function for entire network
	activation_func = tf.nn.relu

	# convolutional layer with sigmoid and max pooling
	# computes 4 features
	conv_layer_1 = conv_layer(x_image, [5, 5, 3, feat_1], [feat_1], activation_func)

	# second convolutional layer with sigmoid and max pooling
	# computes 8 features
	conv_layer_2 = conv_layer(conv_layer_1, [5, 5, feat_1, feat_2], [feat_2], activation_func)

	conv_shape = (32 // 4) ** 2 * feat_2

	# fully connected layer
	fc_layer_1 = fc_layer(conv_layer_2, [conv_shape, fc_neuron], [fc_neuron], [100, conv_shape], activation_func)

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

feat_1_inp = [8, 16, 32]
feat_2_inp = [8, 16, 32]
fc_neuron_inp = [32, 64, 128, 256]
count = 0
max_num = (float('-inf'), {})
for f1 in feat_1_inp:
	for f2 in feat_2_inp:
		for fc in fc_neuron_inp:
			count += 1
			print('\033[92m' + 'running with f1 %d, f2 %d, fc %d on iteration %d' % (f1, f2, fc, count) + '\033[0m')
			max_num = max((run_validation(f1, f2, fc), {'f1': f1, 'f2': f2, 'fc': fc}), max_num, key=lambda x: x[0])

run_net(max_num[1]['f1'], max_num[1]['f2'], max_num[1]['fc'])
print(max_num)
