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

def conv_layer(inp, weight_shape, bias_shape, stride_shape, ksize_shape):
	weights = create_var(weight_shape)
	bias = create_var(bias_shape)
	conv = tf.nn.conv2d(inp, weights, strides=stride_shape, padding='SAME')
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

# declare placeholder variables
x = tf.placeholder(tf.float32, [100, 3072])
keep_prob = tf.placeholder(tf.float32)
y_actual = tf.placeholder(tf.float32, [100, 10])

# reshape into an image with 3 color channels
x_image = tf.reshape(x, [100, 32, 32, 3])

# so can easily change variables for entire network
activation_func = tf.nn.relu
feat_1 = 32
feat_2 = 16
num_conv = 2
conv_shape = ((32 // (num_conv ** 2)) ** 2) * feat_2
fc_neuron = 64
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

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

train_batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
count = 1

for file_name in train_batch:
	data, labels = shuffle_data(unpickle(file_name))
	labels = reshape_labels(labels)
	for k in range(batch_size):
		data_batch = data[k:k + batch_size]
		labels_batch = labels[k:k + batch_size]
		train_step.run(feed_dict={ x: data_batch, y_actual: labels_batch, keep_prob: k_p })
		if k == 99:
			train_accuracy = accuracy.eval(feed_dict={ x: data_batch, y_actual: labels_batch, keep_prob: k_p})
			print("step %d, training accuracy %g"%(count, train_accuracy))
	count += 1

test_data = unpickle('test_batch')
avg = 0
for i in range(100):
	labels = reshape_labels(test_data['labels'])[i:i + 100]
	test_accuracy = accuracy.eval(feed_dict={ x: test_data['data'][i:i+100], y_actual: labels, keep_prob: k_p })
	avg += test_accuracy
	if i % 10 == 0:
		print("test accuracy for batch %d is %g"%(i, test_accuracy))
print("test accuracy overall is " + str(avg / 100.0))

