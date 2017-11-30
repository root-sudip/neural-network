import tensorflow as tf
import numpy as np

from keras.utils import np_utils

import csv



class MLP:

    def MLP_computation(self,d1):
        self.mygraph = tf.Graph()
        print('Computation graph created.')
        with self.mygraph.as_default():
            self.x = tf.placeholder(dtype=tf.float32,shape=[None,4],name="X")
            self.y = tf.placeholder(dtype=tf.float32,shape=[None,3],name="Y")


            w1 = tf.Variable(tf.truncated_normal([4,3],stddev=0.1),name="w1")
            b1 = tf.Variable(tf.constant(0.1,shape=[3]),name="b1")

            w2 = tf.Variable(tf.truncated_normal([3, 5], stddev=0.1), name="w2")
            b2 = tf.Variable(tf.constant(0.1, shape=[5]), name="b2")

            w3 = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1), name="w3")
            b3 = tf.Variable(tf.constant(0.1, shape=[3]), name="b3")


            mul1 = tf.matmul(self.x,w1,name="Multiply")
            h1 = tf.add(mul1,b1,name="addition")
            self.layer1_out = tf.nn.tanh(h1, name="Layer1Activation")

            mul2 = tf.matmul(self.layer1_out, w2, name="Multiply")
            h2 = tf.add(mul2, b2, name="addition")
            self.layer2_out = tf.nn.tanh(h2, name="Layer2Activation")

            mul3 = tf.matmul(self.layer2_out, w3, name="Multiply")
            self.h3 = tf.add(mul3, b3, name="addition")
            self.layer3_out = tf.nn.softmax(self.h3, name="Layer3Activation")

            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.h3))

            self.optimizer=tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

            true=tf.arg_max(self.y,1)
            predicted=tf.arg_max(self.layer3_out,1)

            corrects=tf.cast(tf.equal(true,predicted),tf.float32)
            self.accuracy=tf.reduce_mean(corrects)

            self.init = tf.global_variables_initializer()

            print('Graph Structure Ready.')

    def fit(self,train_x,train_y,epochs,batchsize):
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init])

            nbtrain=len(train_x)
            nbbatches=int(np.ceil(nbtrain/float(batchsize)))


            for ep in range(epochs):

                start=0
                loss=0
                acc=0
                for b in range(nbbatches):
                    end=start+batchsize
                    batchx=train_x[start:end]
                    batchy=train_y[start:end]

                    dict = {self.x: batchx,self.y:batchy}
                    _,bl,ba,lo=sess.run([self.optimizer,self.loss,self.accuracy,self.layer3_out],feed_dict=dict)
                    loss+=bl
                    acc+=ba
                    print(lo)

                loss = loss/nbbatches
                acc = acc/nbbatches
                print('loss : ',loss, 'acc : ',acc)

            #result = sess.run([self.layer3_out], feed_dict=dict)
            #print("result is ", result)


#X = np.arange(40).reshape([10,4])

X_Train = np.zeros([150,4])
Y_Train = np.zeros([150])


i = 0
with open("iris.csv") as fl:
    line = csv.reader(fl)
    for row in line:
        X_Train[i][0] = float(row[0])
        X_Train[i][1] = float(row[1])
        X_Train[i][2] = float(row[2])
        X_Train[i][3] = float(row[3])

        Y_Train[i] = row[3]

        i = i + 1

Y_Train = np_utils.to_categorical(Y_Train)

#print(X_Train)

obj = MLP()
obj.MLP_computation(X_Train)
obj.fit(X_Train,Y_Train,1,32)


