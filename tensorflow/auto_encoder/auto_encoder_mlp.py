import tensorflow as tf
import numpy as np

from keras.utils import np_utils

import csv

from data_representation import load_flatten_data

class MLP:

    def get_layer_shape(self,layer):
        sh=tf.Tensor.get_shape(layer)
        sh=[sh[j].value for j in range(len(sh))]
        return sh

    def flatten_array(self, array):
        shape = array.get_shape().as_list()
        return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

    def MLP_computation(self):
        self.mygraph = tf.Graph()
        print('Computation graph created.')
        with self.mygraph.as_default():

            #i/p and o/p shape
            self.x = tf.placeholder(dtype=tf.float32,shape=[None,1568],name="X")
            self.y = tf.placeholder(dtype=tf.float32,shape=[None,2],name="Y")

            #MLP layer1
            mlp_w1 = tf.Variable(tf.truncated_normal([1568, 32], stddev=0.1), name="mlp_w1", trainable=True)
            mlp_b1 = tf.Variable(tf.constant(0.1, shape=[32]), name="mlp_b1", trainable=True)

            mlp_mul1 = tf.matmul(self.x, mlp_w1, name="Multiply1")
            self.mlp_h1 = tf.add(mlp_mul1, mlp_b1, name="addition1")
            self.mlp_layer1_out = tf.nn.tanh(self.mlp_h1, name="MLPLayer1Activation")
            layer_shape = self.get_layer_shape(self.mlp_layer1_out)
            print('MLP1 : ', layer_shape)
            print("*****************************")

            # MLP layer2
            mlp_w2 = tf.Variable(tf.truncated_normal([32, 64], stddev=0.1), name="mlp_w2", trainable=True)
            mlp_b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="mlp_b2", trainable=True)

            mlp_mul2 = tf.matmul(self.mlp_layer1_out, mlp_w2, name="Multiply2")
            self.mlp_h2 = tf.add(mlp_mul2, mlp_b2, name="addition2")
            self.mlp_layer2_out = tf.nn.tanh(self.mlp_h2, name="MLPLayer2Activation")


            ##drop out

            self.keep_prob = tf.placeholder(dtype=tf.float32, name="drop_out")
            self.drop_out = tf.nn.dropout(self.mlp_layer2_out, self.keep_prob)

            ##dropout end



            layer_shape = self.get_layer_shape(self.mlp_layer2_out)
            print('MLP2 : ', layer_shape)
            print("*****************************")

            # MLP layer3
            mlp_w3 = tf.Variable(tf.truncated_normal([64, 2], stddev=0.1), name="mlp_w3", trainable=True)
            mlp_b3 = tf.Variable(tf.constant(0.1, shape=[2]), name="mlp_b3", trainable=True)

            mlp_mul3 = tf.matmul( self.drop_out, mlp_w3, name="Multiply3")
            self.mlp_h3 = tf.add(mlp_mul3, mlp_b3, name="addition3")
            self.mlp_layer3_out = tf.nn.softmax(self.mlp_h3, name="Layer3Activation")

            layer_shape = self.get_layer_shape(self.mlp_layer3_out)
            print('MLP2 : ', layer_shape)
            print("*****************************")

            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.mlp_h3))

            self.optimizer=tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

            true=tf.arg_max(self.y,1)
            predicted=tf.arg_max(self.mlp_layer3_out,1)

            corrects=tf.cast(tf.equal(true,predicted),tf.float32)
            self.accuracy=tf.reduce_mean(corrects)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            print('Graph Structure Ready.')

    def fit(self,train_x,train_y,epochs,batchsize):
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init])

            nbtrain=len(train_x)
            nbbatches=int(np.ceil(nbtrain/float(batchsize)))

            for ep in range(epochs):
                print('epoch : ',ep)
                start=0
                loss=0
                acc=0
                for b in range(nbbatches):
                    end=start+batchsize
                    batchx=train_x[start:end]
                    batchy=train_y[start:end]

                    dict = {self.x: batchx,self.y:batchy,self.keep_prob:.25}
                    op, bl, ba=sess.run([self.optimizer,self.loss,self.accuracy],feed_dict=dict)
                    loss+=bl
                    acc+=ba
                    #print(lo)
                self.saver.save(sess, "Weights/last")
                loss = loss/nbbatches
                acc = acc/nbbatches
                print('loss : ',loss, 'acc : ',acc)


            #result = sess.run([self.layer3_out], feed_dict=dict)
            #print("result is ", result)

    def predict(self, X_train, Y_train,batchsize):
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init])
            new_saver = tf.train.import_meta_graph('Weights/last.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('Weights/'))
            start = 0
            loss = 0
            acc = 0

            nbtrain = len(X_train)
            nbbatches = int(np.ceil(nbtrain / float(batchsize)))

            for b in range(nbbatches):

                print("Reading bacth : ",b)
                end = start + batchsize
                batchx = X_train[start:end]
                batchy = Y_train[start:end]

                dict = {self.x: batchx, self.y: batchy, self.keep_prob:.25}
                op, bl, ba = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=dict)
                loss += bl
                acc += ba
                # print(lo)
            loss = loss / nbbatches
            acc = acc / nbbatches
            print('loss : ', loss, 'acc : ', acc)






obj = MLP()
obj.MLP_computation()

inp = input('Training(Train)/Testing(Test)?')

if inp == 'Train':
    X_train, Y_train = load_flatten_data('train_flatten.csv', 287961, 1568)
    obj.fit(X_train,Y_train,20,128)
elif inp == 'Test':
    X_train, Y_train = load_flatten_data('test_flatten.csv', 48081, 1568)
    obj.predict(X_train,Y_train,128)
else:
    print('You should write Train/Test.')
