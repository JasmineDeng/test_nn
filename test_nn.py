import numpy as np
import tensorflow as tf
import random

FLAGS = None

# (0.5360999712347985, {'f1': 64, 'f2': 256, 'fc': 2048})

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

def conv_layer(name, inp, weight_shape, bias_shape, act_func, tb=False):
	with tf.name_scope(name):
		with tf.name_scope('weights'):
			weights = create_var(weight_shape)
			if tb:
				tf.histogram_summary(name + ' weights', weights)
		with tf.name_scope('bias'):
			bias = create_var(bias_shape)
			if tb:
				tf.histogram_summary(name + ' bias', bias)
		with tf.name_scope('conv'):
			conv = tf.nn.conv2d(inp, weights, strides=[1, 1, 1, 1], padding='SAME')
			pre_activate = conv + bias
		activation = act_func(pre_activate, name='activation')
		pool = tf.nn.max_pool(activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling')
		norm = tf.nn.lrn(pool, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
		return norm

def fc_layer(name, inp, weight_shape, bias_shape, reshape_shape, act_func, tb=False):
	with tf.name_scope(name):
		with tf.name_scope('weights'):
			weights = create_var(weight_shape)
			if tb:
				tf.histogram_summary(name + ' weights', weights)
		with tf.name_scope('bias'):
			bias = create_var(bias_shape)
			if tb:
				tf.histogram_summary(name + ' bias', bias)
		with tf.name_scope('Wx_plus_b'):
			new_inp = tf.reshape(inp, reshape_shape)
			pre_activate = tf.matmul(new_inp, weights) + bias
		activation = act_func(pre_activate, name='activation')
		return activation

def run_net(feat_1, feat_2, fc_neuron, train_batch=['data_batch_1', 'data_batch_2', 'data_batch_3', \
	'data_batch_4', 'data_batch_5'], test_batch='test_batch', print_acc=True, normalize=False, path='log', tensorboard=False):

	x = tf.placeholder(tf.float32, [100, 3072])
	y_actual = tf.placeholder(tf.float32, [100, 10])
	x_image = tf.reshape(x, [100, 32, 32, 3])

	# so can easily change activation function for entire network
	activation_func = tf.nn.relu

	# convolutional layer with sigmoid and max pooling
	# computes 4 features
	conv_layer_1 = conv_layer('conv_layer_1', x_image, [5, 5, 3, feat_1], [feat_1], activation_func, tb=tensorboard)

	# second convolutional layer with sigmoid and max pooling
	# computes 8 features
	conv_layer_2 = conv_layer('conv_layer_2', conv_layer_1, [5, 5, feat_1, feat_2], [feat_2], activation_func, tb=tensorboard)

	conv_shape = (32 // 4) ** 2 * feat_2

	# fully connected layer
	fc_layer_1 = fc_layer('fc_layer_1', conv_layer_2, [conv_shape, fc_neuron], 
		[fc_neuron], [100, conv_shape], activation_func, tb=tensorboard)

	keep_prob = tf.placeholder(tf.float32)
	dropout_1 = tf.nn.dropout(fc_layer_1, keep_prob)

	# reshape into 10 
	weight_reshape = create_var([fc_neuron, 10])
	bias_reshape = create_var([10])
	max_pool = tf.reshape(dropout_1, [100, fc_neuron])

	y_result = tf.matmul(max_pool, weight_reshape) + bias_reshape
	
	with tf.name_scope('cross_entropy'):
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_result)
		with tf.name_scope('total'):
			cost = tf.reduce_mean(diff)
	if tensorboard:
		tf.scalar_summary('cross_entropy', cost)
	
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

	with tf.name_scope('accuracy_graph'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_result, 1), tf.argmax(y_actual, 1))
		with tf.name_scope('accuracy'):	
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	if tensorboard:
		acc1_summary = tf.scalar_summary('training accuracy', accuracy)

	with tf.name_scope('gradient'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_result))
		grads = tf.gradients(loss, tf.trainable_variables())
		grads = list(zip(grads, tf.trainable_variables()))
		for grad, var in grads:
			if grad is not None and tensorboard:
				tf.histogram_summary(var.name + '/gradient', grad)

	sess = tf.InteractiveSession()
	if tensorboard:
		summary_op = tf.merge_all_summaries()
		train_writer = tf.train.SummaryWriter(path, graph=tf.get_default_graph())
		acc2_summary = tf.scalar_summary('test accuracy', accuracy)
		acc3_summary = tf.scalar_summary('final accuracy', accuracy)
	sess.run(tf.initialize_all_variables())

	cifar_data = {'data': [], 'labels': []}
	for name in train_batch:
		new_data = unpickle(name)
		cifar_data['data'].extend(new_data['data'])
		cifar_data['labels'].extend(new_data['labels'])
	cifar_data = shuffle_data(cifar_data)

	test_data = unpickle(test_batch)
	test_data = shuffle_data(test_data)

	for i in range(len(cifar_data[0]) // 100):
		data = cifar_data[0][100 * i: 100 * i + 100]
		labels = cifar_data[1][100 * i: 100 * i + 100]
		if normalize:
			normalize_rgb(data)
		labels = reshape_labels(labels)
		feed = { x: data, y_actual: labels, keep_prob: 0.5}
		if tensorboard:
			_, summary = sess.run([train_step, summary_op], feed_dict=feed)
			train_writer.add_summary(summary, i * 100)
			if i % 10 == 0:
				feed_test = { x: test_data[0][i: i + 100], y_actual: reshape_labels(test_data[1][i:i+100]), keep_prob: 0.5 }
				test_accuracy = sess.run(acc2_summary, feed_dict=feed_test)
				test_data = shuffle_data({'data': test_data[0], 'labels': test_data[1]})
				train_writer.add_summary(test_accuracy, i * 100)
		else:
			train_step.run(feed_dict=feed)
		if i % 100 == 0 and print_acc:
			train_accuracy = accuracy.eval(feed_dict=feed)
			print("step %d, training accuracy %g"%(i, train_accuracy))

	avg = 0
	for i in range(100):
		labels = reshape_labels(test_data[1])[i:i + 100]
		feed = { x: test_data[0][i:i+100], y_actual: labels, keep_prob: 0.5 }
		if tensorboard:
			acc = sess.run(acc3_summary, feed_dict=feed)
			train_writer.add_summary(acc, i)
		test_accuracy = accuracy.eval(feed_dict=feed)
		avg += test_accuracy
		if i % 10 == 0 and print_acc:
			print("test accuracy for batch %d is %g"%(i, test_accuracy))
	print(GREEN + "test accuracy overall is " + str(avg / 100.0) + END)
	return float(avg) / 100.0

def run_cross_validation(feat_1_inp, feat_2_inp, fc_neuron_inp, tb=False):
	count = 0
	max_num = (float('-inf'), {})
	train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
	test_batch = 'data_batch_5'
	for f1 in feat_1_inp:
		for f2 in feat_2_inp:
			for fc in fc_neuron_inp:
				count += 1
				print(GREEN + 'running with f1 %d, f2 %d, fc %d on iteration %d' % (f1, f2, fc, count) + END)
				acc = run_net(f1, f2, fc, train_batch, test_batch, print_acc=False, tensorboard=tb)
				max_num = max((acc, {'f1': f1, 'f2': f2, 'fc': fc}), max_num, key=lambda x: x[0])
				print(GREEN + 'current max ' + str(max_num) + END)

	train_batch.append('data_batch_5')
	run_net(max_num[1]['f1'], max_num[1]['f2'], max_num[1]['fc'], train_batch, 'test_batch')
	print(max_num)

GREEN = '\033[92m'
END = '\033[0m'
# run net
# run_net(16, 32, 128)
