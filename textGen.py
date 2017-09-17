import tensorflow as tf
import numpy as np
import zipfile
import collections
import random
import math
import string
filename = 'text8.zip'

#read in the text corpus
def readfile(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

text = readfile(filename)
print ('Text has been read in')

class BatchGen(object):
        def __init__(self, text, batch_size, num_unfolds):
                self._text = text
                self._text_size = len(self._text)
                self._batch_size = batch_size
                self._num_unfolds = num_unfolds
		self._data, self._count, self._dict, self._reverse_dict =  self.build_dataset()              

		segment = self._text_size // (batch_size)
                self._cursor = [ offset * segment for offset in range(batch_size + 1)]
                self._last_batch = self._batch()

	#generate a dictionary and reverse dictionary of words from the text corpus
	def build_dataset(self):
		count = [['UNK', -1]]
		count.extend(collections.Counter(self._text).most_common(vocabulary_size - 1))
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)
		data = list()
		unk_count = 0
		for word in self._text:
			if word in dictionary:
				index = dictionary[word]
			else:
				index = 0  # dictionary['UNK']
				unk_count = unk_count + 1
			data.append(index)
		count[0][1] = unk_count
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return data, count, dictionary, reverse_dictionary
	
	def _batch(self):
		words = np.zeros(shape = (batch_size), dtype = np.int32)
                for index in range(self._batch_size):
                        words[index] =  self._data[self._cursor[index]]
                        self._cursor[index] = (self._cursor[index] + 1) % self._text_size
                return words

        def next_batch(self):
                batch = [self._last_batch]
                for i in range(self._num_unfolds):
                        batch.append(self._batch())
                self._last_batch = batch[-1]
                return batch

def logprob(labels, predic):
	predic[predic < 1*10**(-10)] = 1*10**(-10)
	h = np.sum(np.multiply(labels,-np.log(predic))) / labels.shape[0]
	return h

#generate a random normalized probability distribution
def random_distribution():
	b = np.random.uniform(0.0, 1.0, size = [1,vocabulary_size])
        return b/np.sum(b,1)[0]

#pick the most likely next word based on the continuous distribution function of the pdf
def cdf(prob_dis):
	random_threshold = random.uniform(0,1)
	cont_prob = 0
	for i in range(len(prob_dis)):
		cont_prob += prob_dis[i]
		if cont_prob >= random_threshold:
			return i
	return len(prob_dis) - 1

def word_trans(prob_dis):
	return [train_total._reverse_dict[word] for word in prob_dis]

def one_hot_gen(batch):
        new_batch = list()
        for index in range(len(batch)):
                hot = np.zeros([vocabulary_size])
                hot[(batch[index])] = 1
                new_batch.append(hot)
        return new_batch


#hyper-params
vocabulary_size = 25000
batch_size = 64
num_unfolds = 10
hidden = 100

alpha = 10.0
clip_norm = 1.25
dSteps = 150
dRate = 0.9
beta = 0.01
dropout = 0.75

num_layers = 2
num_embed = 100
nodes = 25

train_total = BatchGen(text, batch_size, num_unfolds)
del text

graph = tf.Graph()

with graph.as_default():

	#define lstm cell
        def lstm(wx, wm, ib, fb, cb, ob, i, o, state):
		i = tf.nn.dropout(i, keep_prob)
		
               	#weight matrices
                wi = tf.matmul(i,wx)
                wo = tf.matmul(o,wm)

                #memory cell processes with peepholes
                input_gate = tf.sigmoid(wi[:, 0:hidden] + wo[:, 0:hidden] + ib + state)
                forget_gate = tf.sigmoid(wi[:, hidden:2*hidden] + wo[:, hidden:2*hidden] + state)
                update = tf.tanh(wi[:, 2*hidden:3*hidden] + wo[:, 2*hidden:3*hidden] + cb)
                state = forget_gate*state + input_gate*update
                output_gate = tf.sigmoid(wi[:, 3*hidden:4*hidden] + wo[:, 3*hidden:4*hidden] + ob + state)
                output = output_gate * tf.tanh(state)

		return output, state

	#define an individual lstm layer run (allows for stacking lstms)
	def layer(layer_num, unfolds, batch_size, batch, save_state, save_output):	
		num = layer_num - 1
		start = num*hidden
		end = (num+1)*hidden
		output = save_output[:, start:end]
		state = save_state[:, start:end]
		outputs = list()
		
		for i in range(unfolds):
			output, state = lstm(wx[:, 4*start:4*end],wm[:, 4*start:4*end], ib[start:end], fb[start:end], 
							cb[start:end], ob[start:end], batch[i], output, state)
			outputs.append(output)

		
		return outputs, output, state

	def model(batch, batch_size, save_output, save_state, train = False):	
		global num_unfolds

		if train == False:
			unfolds = 1
			batch = [batch]
		else:
			unfolds = num_unfolds
		
		embed = tf.nn.embedding_lookup(embeddings,batch)
		states = list()
		outputs = list()
		for layers in range(1,(num_layers + 1)):
			if layers == 1: 
				layer_batch = embed
			elif layers > 1:
				layer_batch = logit
				print (logit)
			

			logit, output, state = layer(layers, unfolds, batch_size, layer_batch, save_state, save_output)
			states.append(state)
			outputs.append(output)

		#classifier	
		logit = tf.concat(logit,0)
		logits_1 = tf.nn.xw_plus_b(logit, w_log1, b_log1)
		logits_2 = tf.nn.xw_plus_b(logits_1, w_log2, b_log2)		

		return logits_2, outputs, states

	

	#lstm weights
	wx = tf.Variable(tf.truncated_normal([num_embed, num_layers*(4*hidden)]), name='wx')
        wm = tf.Variable(tf.truncated_normal([hidden, num_layers*(4*hidden)]), name = 'wm')
	ws = tf.Variable(tf.truncated_normal([3*hidden, num_layers*hidden])) #3 gates


	#lstm biases
        ib = tf.Variable(tf.zeros([num_layers*hidden]), name='bias_1')
	fb = tf.Variable(tf.zeros([num_layers*hidden]), name='bias_2') 
	cb = tf.Variable(tf.zeros([num_layers*hidden]), name='bias_3') 
	ob = tf.Variable(tf.zeros([num_layers*hidden]), name='bias_4') 

	#saved states and output
        save_state = tf.Variable(tf.zeros(shape=[batch_size, num_layers*hidden]),  trainable = False, name = 'save_state' )
        save_output = tf.Variable(tf.zeros(shape=[batch_size, num_layers*hidden]),  trainable = False, name = 'save_output')

	#classifier weights and biases       
	w_log1 = tf.Variable(tf.truncated_normal([hidden, nodes]))
	b_log1 = tf.Variable( tf.zeros([nodes]))
	w_log2 = tf.Variable(tf.truncated_normal([nodes, vocabulary_size]))
	b_log2 = tf.Variable(tf.zeros([vocabulary_size]))

 	global_step = tf.Variable(0)

	embeddings = tf.placeholder(tf.float32, [vocabulary_size, num_embed])

	#Load training data
	train_batch = list()
        for i in range(num_unfolds):
                train_batch.append(
                               tf.placeholder(shape = [batch_size], dtype = tf.int32))
	train_labels = tf.placeholder(shape = [num_unfolds, batch_size, vocabulary_size], dtype = tf.int32)
	keep_prob = tf.placeholder(tf.float32)

	#run the model
	logits, outputs, states = model(train_batch, batch_size, save_output, save_state, True)

	#save the states and outputs of the lstms and then run loss
	states = tf.reshape(states, [batch_size, num_layers*hidden])
	outputs = tf.reshape(outputs, [batch_size, num_layers*hidden])
        with tf.control_dependencies([save_output.assign(outputs), save_state.assign(states)]):

		loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(train_labels, 0), logits=logits))

		loss = 	loss + beta*sum( 
					tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
                                	if not ('bias' in tf_var.name or 'b_' in tf_var.name or 'Variable' in tf_var.name))


        #gradient clipping
      	learn_rate = tf.train.exponential_decay(alpha, global_step, dSteps, dRate, staircase = True )
      	optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        gradients, var = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        optimizer = optimizer.apply_gradients(zip(gradients,var), global_step=global_step)

	train_prediction = tf.nn.softmax(logits)

	#sampling eval: batch 1, no unrolling.
	sample_input = tf.placeholder(tf.int32, shape=[1])

	saved_sample_output = tf.Variable(tf.zeros([1, num_layers*hidden]))
	saved_sample_state = tf.Variable(tf.zeros([1, num_layers*hidden]))

	reset_sample_state = tf.group(
		saved_sample_output.assign(tf.zeros([1, num_layers*hidden])),
	  	saved_sample_state.assign(tf.zeros([1, num_layers*hidden])))

	logits, outputs, states = model(sample_input, 1, saved_sample_output, saved_sample_state)

	states = tf.reshape(states, [1, num_layers*hidden])
	outputs = tf.reshape(outputs, [1, num_layers*hidden])
	with tf.control_dependencies([saved_sample_output.assign(outputs), saved_sample_state.assign(states)]):
		sample_prediction = tf.nn.softmax(logits)


with tf.Session(graph = graph) as session:
	tf.global_variables_initializer().run()

	#load in the embedding space generated using a skip-gram model
 	saver = tf.train.import_meta_graph('model/skip_gram.meta')
        saver.restore(session, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        embed = graph.get_tensor_by_name('skip_embeddings:0')
        embed = session.run(embed)

	num_steps = 5001
	average = 0
	for step in range(num_steps):
		feed = dict()
		feed[keep_prob] = dropout
		feed[embeddings] = embed
	
		batch = train_total.next_batch()

		labels = list()
		for i in range(num_unfolds):
			labels.append(one_hot_gen(batch[i+1]))

		feed[train_labels] = labels
	
		for i in range(num_unfolds):
			feed[train_batch[i]] = batch[i]

		_, l, predic = session.run([optimizer, loss, train_prediction], feed_dict = feed)	

		average += l
                if step % 20 == 0:
			perplexity = np.exp(logprob(np.reshape(labels, (num_unfolds*batch_size, vocabulary_size)),predic))
                        print (average / 21.0, step, perplexity)
			average = 0

                if step % 50 == 0:
                        #generate some samples
                        print('=' * 80)
                        for _ in range(5):
                                feed = [cdf(random_distribution()[0])]
                                sentence = word_trans(feed)[0]
                                reset_sample_state.run()
                                for i in range(20):
					prediction = sample_prediction.eval({sample_input: feed, keep_prob: 1, embeddings: embed})
                                        feed = [cdf(prediction[0])]
					sentence += ' ' +   word_trans(feed)[0]
				print (sentence)	
                        print('=' * 80)
