import tensorflow as tf
import numpy as np 


class LSTMLayer(object):    
	def __init__(self, sess,
					   num_units, 
					   hidden_layer_size,
					   output_classes,
					   embedding_dim,
					   epoch,
					   optimizer,
					   learning_rate,
					   batch_size,
					   loss,
					   eval_metric,
					   reg_lambda,
					   init_std,
					   vocab_size,
					   bidirectional=True,
					   verbose=True       
				):
		
		self.sess = sess
		
		self.lstm_params = {}
		self.lstm_params['num_units'] = num_units
		self.lstm_params['hidden_layer_size'] = hidden_layer_size
		self.lstm_params['output_classes'] = output_classes
		self.lstm_params['bidirectional'] = bidirectional
		self.lstm_params['embedding_dim'] = embedding_dim
		
		self.training_params = {}
		self.training_params['epoch'] = epoch
		self.training_params['optimizer'] = optimizer
		self.training_params['learning_rate'] = learning_rate
		self.training_params['batch_size'] = batch_size

		self.training_params['loss'] = loss
		self.training_params['reg_lambda'] = reg_lambda
		self.training_params['init_std'] = init_std
		
		self.other_params = {}
		self.other_params['vocab_size'] = vocab_size
		self.other_params['verbose'] = verbose


		
		
		#self.__init_weights()
		#self.__construct_graph()
	
	def load_weights(self, pretrained_weights = None, word_embeddings = None):
		bi = self.lstm_params['bidirectional']
		v_s = self.other_params['vocab_size']
		e = self.lstm_params['embedding_dim']
		init_std = self.training_params['init_std']

		if pretrained_weights is None:
			self.__init_weights()
		else:
			self.left_encoder_weights = pretrained_weights['left_encoder_weights']
			if bi:
				self.right_encoder_weights = pretrained_weights['right_encoder_weights']
			self.output_weights = pretrained_weights['output_weights'] 
		
			self.convert_to_tensors()
		
		if word_embeddings is None:
			self.word_embeddings_matrix = tf.Variable(tf.random_normal(shape=(v_s,e),stddev=init_std,dtype=tf.float64),trainable=True, dtype=tf.float64)
		else:
			self.word_embeddings_matrix = tf.Variable(word_embeddings,trainable=False,dtype=tf.float64)
		
	def __init_weights(self):
		
		#Retrive params
		d = self.lstm_params['hidden_layer_size']
		e = self.lstm_params['embedding_dim']
		c = self.lstm_params['output_classes']
		bi =  self.lstm_params['bidirectional']
		init_std = self.training_params['init_std']

		#Init params
		self.left_encoder_weights = {}
		self.left_encoder_weights['Wxh_Left'] = tf.Variable(tf.random_normal(shape = (4*d,e),stddev=init_std,dtype=tf.float64), name = 'Wxh_Left',dtype=tf.float64)
		self.left_encoder_weights['bxh_Left'] = tf.Variable(tf.random_normal(shape = (4*d,1),stddev=init_std,dtype=tf.float64), name = 'bxh_Left',dtype=tf.float64)
		self.left_encoder_weights['Whh_Left'] = tf.Variable(tf.random_normal(shape = (4*d,d),stddev=init_std,dtype=tf.float64), name = 'Whh_Left',dtype=tf.float64)
		self.left_encoder_weights['bhh_Left'] = tf.Variable(tf.random_normal(shape = (4*d,1),stddev=init_std,dtype=tf.float64), name = 'bhh_Left',dtype=tf.float64)
		
		if bi:
			self.right_encoder_weights = {}
			self.right_encoder_weights['Wxh_Right'] = tf.Variable(tf.random_normal(shape = (4*d,e),stddev=init_std,dtype=tf.float64), name = 'Wxh_Right',dtype=tf.float64)
			self.right_encoder_weights['bxh_Right'] = tf.Variable(tf.random_normal(shape = (4*d,1),stddev=init_std,dtype=tf.float64), name = 'bxh_Right',dtype=tf.float64)
			self.right_encoder_weights['Whh_Right'] = tf.Variable(tf.random_normal(shape = (4*d,d),stddev=init_std,dtype=tf.float64), name = 'Whh_Right',dtype=tf.float64)
			self.right_encoder_weights['bhh_Right'] = tf.Variable(tf.random_normal(shape = (4*d,1),stddev=init_std,dtype=tf.float64), name = 'bhh_Right',dtype=tf.float64)
			
		# softmax
		self.output_weights = {}
		self.output_weights['Why_Left'] = tf.Variable(tf.random_normal(shape = (c,d),stddev=init_std,dtype=tf.float64), name = "Why_Left",dtype=tf.float64)
		if bi:
			self.output_weights['Why_Right'] = tf.Variable(tf.random_normal(shape = (c,d),stddev=init_std,dtype=tf.float64),name = "Why_Right",dtype=tf.float64)

	def convert_to_tensors(self):
		for key,value in self.left_encoder_weights.items():
			self.left_encoder_weights[key] = tf.Variable(value, name=key)
		
		if self.lstm_params['bidirectional']:
			for key,value in self.right_encoder_weights.items():
				self.right_encoder_weights[key] = tf.Variable(value, name=key)
			
		for key,value in self.output_weights.items():
			self.output_weights[key] = tf.Variable(value, name=key)
			

	def construct_graph(self):
		T = self.lstm_params['num_units']
		d = self.lstm_params['hidden_layer_size']
		bi = self.lstm_params['bidirectional']
		e = self.lstm_params['embedding_dim']
		b_s = self.training_params['batch_size']
		lbda = self.training_params['reg_lambda'] 
		classes = self.lstm_params['output_classes']
		
		

		#weights
		Wxh_Left = self.left_encoder_weights['Wxh_Left']
		bxh_Left = self.left_encoder_weights['bxh_Left']
		Whh_Left = self.left_encoder_weights['Whh_Left']
		bhh_Left = self.left_encoder_weights['bhh_Left']
		Why_Left = self.output_weights['Why_Left']
		
		self.idxs = tf.placeholder(dtype=tf.int32,shape = (None, T), name = 'word_indecies') #None*T
		self.y = tf.placeholder(dtype=tf.int32, shape = (None, classes), name = 'labels')#None*1
		self.batch_size = tf.placeholder(dtype=tf.int32,name = 'batch_size')


		
		self.word_embd_lookup = tf.nn.embedding_lookup(self.word_embeddings_matrix, self.idxs,name='embedding_lookup')# None*T*e

		shp = tf.stack([self.batch_size,tf.constant(d,dtype=tf.int32)],axis=0)
		self.h_Left = [None]*(T+1)
		self.h_Left[-1] = tf.zeros(shape=shp, dtype=tf.float64)
		self.c_Left = [None]*(T+1)
		self.c_Left[-1] = tf.zeros(shape=shp, dtype=tf.float64)
		
		self.gates_xh_Left = [None]*T
		self.gates_hh_Left = [None]*T
		self.gates_pre_Left = [None]*T  
		self.gates_Left = [None]*T
		
		if bi:
			self.gates_xh_Right = [None]*T
			self.gates_hh_Right = [None]*T
			self.gates_pre_Right = [None]*T
			self.gates_Right = [None]*T

			self.h_Right = [None]*(T+1)
			self.h_Right[-1] = tf.zeros(shape=shp, dtype=tf.float64)
			self.c_Right = [None]*(T+1)
			self.c_Right[-1] = tf.zeros(shape=shp, dtype=tf.float64)
			
			#Retrive weights
			Wxh_Right = self.right_encoder_weights['Wxh_Right']
			bxh_Right = self.right_encoder_weights['bxh_Right']
			Whh_Right = self.right_encoder_weights['Whh_Right']
			bhh_Right = self.right_encoder_weights['bhh_Right']
			Why_Right = self.output_weights['Why_Right']


			
			self.word_embd_lookup_rev = tf.manip.reverse(self.word_embd_lookup, axis = [-2], name = "embedding_lookup_rev") #None*T*e


		for t in range(T):
			
			a = tf.reduce_sum(tf.slice(self.word_embd_lookup,[0,t,0],[-1,1,-1]),axis =1)
			self.gates_xh_Left[t] = tf.matmul(a, tf.transpose(Wxh_Left)) + tf.reshape(bxh_Left,shape=(1,-1))
			self.gates_hh_Left[t] = tf.matmul(self.h_Left[t-1] , tf.transpose(Whh_Left)) + tf.reshape(bhh_Left,shape=(1,-1))
			self.gates_pre_Left[t] =  self.gates_xh_Left[t] + self.gates_hh_Left[t]
			
			i = tf.sigmoid(tf.slice(self.gates_pre_Left[t], [0,0], [-1,d]))
			f = tf.sigmoid(tf.slice(self.gates_pre_Left[t], [0,2*d], [-1,d]))
			o = tf.sigmoid(tf.slice(self.gates_pre_Left[t], [0,3*d], [-1,d]))
			g = tf.tanh(tf.slice(self.gates_pre_Left[t], [0,d], [-1,d]))
			self.gates_Left[t] = tf.concat([i,g,f,o],axis = -1)
			
			i = tf.slice(self.gates_Left[t], [0,0], [-1,d])
			g = tf.slice(self.gates_Left[t], [0,d], [-1,d])
			f = tf.slice(self.gates_Left[t], [0,2*d], [-1,d])
			o = tf.slice(self.gates_Left[t], [0,3*d], [-1,d])
			
			self.c_Left[t]  = f*self.c_Left[t-1] + i*g
			self.h_Left[t]  = o*tf.tanh(self.c_Left[t])
			
			if bi:
				a_rev = tf.reduce_sum(tf.slice(self.word_embd_lookup_rev,[0,t,0],[-1,1,-1]),axis =1)
				self.gates_xh_Right[t] = tf.matmul(a_rev, tf.transpose(Wxh_Right)) + tf.reshape(bxh_Right,shape=(1,-1))
				self.gates_hh_Right[t]  = tf.matmul(self.h_Right[t-1] , tf.transpose(Whh_Right)) + tf.reshape(bhh_Right,shape=(1,-1))
				self.gates_pre_Right[t] =  self.gates_xh_Right[t] + self.gates_hh_Right[t]
				
				i = tf.sigmoid(tf.slice(self.gates_pre_Right[t], [0,0], [-1,d]))
				f = tf.sigmoid(tf.slice(self.gates_pre_Right[t], [0,2*d], [-1,d]))
				o = tf.sigmoid(tf.slice(self.gates_pre_Right[t], [0,3*d], [-1,d]))
				g = tf.tanh(tf.slice(self.gates_pre_Right[t], [0,d], [-1,d]))
				self.gates_Right[t] = tf.concat([i,g,f,o],axis = -1)
	
				
				i = tf.slice(self.gates_Right[t], [0,0], [-1,d])
				g = tf.slice(self.gates_Right[t], [0,d], [-1,d])
				f = tf.slice(self.gates_Right[t], [0,2*d], [-1,d])
				o = tf.slice(self.gates_Right[t], [0,3*d], [-1,d])
				
				self.c_Right[t]  = f*self.c_Right[t-1] + i*g
				self.h_Right[t]  = o*tf.tanh(self.c_Right[t])
				


		self.y_Left  = tf.matmul(self.h_Left[T-1], tf.transpose(Why_Left))
		self.s = self.y_Left
		if bi:
			self.y_Right = tf.matmul(self.h_Right[T-1], tf.transpose(Why_Right))
			self.s       = self.s + self.y_Right
			
		
		self.sparse_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.s,labels=self.y))
		cost = self.sparse_cross_entropy + lbda*tf.nn.l2_loss(self.word_embeddings_matrix)
		self.optimizer = tf.train.AdamOptimizer(self.training_params['learning_rate']).minimize(cost)
		
		correct_prediction = tf.equal(tf.argmax(self.s,1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
	
	
	def train(self, seqs, labels,val_seqs=None,val_labels=None):
		
		epoch = self.training_params['epoch'] 
		verbose = self.other_params['verbose']
		
		for i in range(epoch):
			print('*'*120)
			start = time()

			for b_x, b_y in self.gen_batches(seqs, labels):
				self.sess.run([self.optimizer],feed_dict = {
											 self.idxs:b_x,
											 self.y:b_y,
											 self.batch_size:b_x.shape[0]
							  })
			
			
			if verbose>0:
				train_acc,train_loss  = self.evaluate(seqs, labels)
				print('[{}] train acc: {:>6}%  train loss: {:>6}'.format(i,train_acc,train_loss))
				
				if val_seqs is not None:
					val_acc,val_loss  = self.evaluate(val_seqs, val_labels)
					print('[{}] val   acc: {:>6}%  val   loss: {:>6}'.format(i,val_acc,val_loss))    
				print('time: {}'.format(time()- start))
	 
	
	def gen_batches(self, seqs, labels=None):
        N = seqs.shape[0]
        
        b_s = self.training_params['batch_size']
        
        for i in range(0,N,b_s):
            idxces = range(i,i+b_s)
            if i+b_s >N:
                if labels is not None:
                    yield seqs[i:],labels[i:]
                else:
                    yield seqs[i:]
            
            else:
                if labels is not None:
                    yield seqs[idxces],labels[idxces]
                else:
                    yield seqs[idxces]
	
	def predict_scores(self, w):
        out = []
        w = np.array(w).reshape(-1, self.lstm_params['num_units'])
        for b_x in self.gen_batches(w):
            
            scores = self.sess.run([self.s],feed_dict={
                        self.idxs: b_x,
                        self.batch_size: b_x.shape[0]
                     })
            out.append(scores)
        return np.vstack((out))
	
	def evaluate(self, data, labels):
	  
	  total_loss = 0
	  total_acc = 0
	  N = data.shape[0]
	  
	  for b_x, b_y in self.gen_batches(data, labels):
		acc, loss =  self.sess.run([self.accuracy,self.sparse_cross_entropy],feed_dict = {
										   self.idxs:b_x,
										   self.y:b_y,
										   self.batch_size:b_x.shape[0]
							})
		total_acc += b_x.shape[0] * acc
		total_loss += b_x.shape[0] * loss
	  
	  
	  return  total_acc/N, total_loss/N


	  