import tensorflow as tf
import numpy as np


from data_representation import load_MNIST


class CNN:
    def __init__(self,CONV,POOL,DENSE):
        #pass

        self.INPUT_SHAPE = [None, 28, 28, 1]
        self.OUTPUT_SHAPE = [None, 10]

        self.CONV1_F = [5, 5, 1, CONV[0]]
        self.CONV1_S = [1, 1, 1, 1]
        self.CONV1_POOL = [1, POOL[0], POOL[0], 1]
        self.CONV1_POOL_S = [1, 1, 1, 1]

        self.CONV2_F = [5, 5, 8, CONV[1]]
        self.CONV2_S = [1, 1, 1, 1]
        self.CONV2_POOL = [1,POOL[1], POOL[1], 1]
        self.CONV2_POOL_S = [1, 2, 2, 1]

        self.CONV3_F = [5, 5, 16,CONV[2]]
        self.CONV3_S = [1, 1, 1, 1]
        self.CONV3_POOL = [1, POOL[2], POOL[2],1]
        self.CONV3_POOL_S = [1, 2, 2, 1]

        self.D1_NODE = DENSE[0]
        self.D2_NODE = DENSE[1]

    def get_layer_shape(self,layer):
        sh=tf.Tensor.get_shape(layer)
        sh=[sh[j].value for j in range(len(sh))]
        return sh

    def flatten_array(self, array):
        shape = array.get_shape().as_list()
        return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

    def CNN_computation(self):
        self.mygraph = tf.Graph()
        print('Computation graph created.')
        with self.mygraph.as_default():

            self.x = tf.placeholder(dtype = tf.float32, shape = self.INPUT_SHAPE, name = "X")  #network input shape
            self.y = tf.placeholder(dtype = tf.float32, shape = self.OUTPUT_SHAPE , name = "Y") #output shape

            # CONV1
            w1 = tf.Variable(tf.truncated_normal(self.CONV1_F, stddev=0.1),name = "W1")
            b1 = tf.Variable(tf.zeros([8]),name="b1")

            layer1_conv = tf.nn.conv2d(self.x, w1, self.CONV1_S, padding = 'SAME',name = "C1_layer")

            layer_shape = self.get_layer_shape(layer1_conv)
            print('CONV1 : ', layer_shape)

            layer1_activaion = tf.nn.sigmoid(layer1_conv + b1,name = "l1_activation")
            self.layer1_pool = tf.nn.max_pool(layer1_activaion, self.CONV1_POOL, self.CONV1_POOL_S, padding = 'SAME',name = "l1_pool")
            layer_shape=self.get_layer_shape(self.layer1_pool)
            print('POOL1 : ',layer_shape)
            #END

            # conv2
            w2 = tf.Variable(tf.truncated_normal( self.CONV2_F, stddev=0.1), name = "W2")
            b2 = tf.Variable(tf.zeros([16]), name="b1")

            layer2_conv = tf.nn.conv2d(self.layer1_pool, w2, self.CONV2_S, padding = 'SAME', name = "C2_layer")

            layer_shape = self.get_layer_shape(layer2_conv)
            print('CONV2 : ', layer_shape)

            layer2_activaion = tf.nn.sigmoid(layer2_conv + b2, name = "l2_activation")
            self.layer2_pool = tf.nn.max_pool(layer2_activaion, self.CONV2_POOL, self.CONV2_POOL_S, padding = 'SAME',name = "l2_pool")
            layer_shape = self.get_layer_shape(self.layer2_pool)
            print('POOL2 : ', layer_shape)
            #END

            # conv3
            w3 = tf.Variable(tf.truncated_normal(self.CONV3_F, stddev=0.1), name = "W3")
            b3 = tf.Variable(tf.zeros([32]), name = "b3")

            layer3_conv = tf.nn.conv2d(self.layer2_pool, w3, self.CONV3_S, padding = 'SAME', name = "C3_layer")

            layer_shape = self.get_layer_shape(layer3_conv)
            print('CONV3 : ', layer_shape)

            layer3_activaion = tf.nn.sigmoid(layer3_conv + b3, name = "l3_activation")
            self.layer3_pool = tf.nn.max_pool(layer3_activaion, self.CONV3_POOL, self.CONV3_POOL_S, padding = 'SAME',name = "l3_pool")
            layer_shape = self.get_layer_shape(self.layer3_pool)
            print('POOL3 : ', layer_shape)
            #END

            flatten_dim = layer_shape[-1]*layer_shape[-2]*layer_shape[-3]

            self.Flatten=tf.reshape(self.layer3_pool,[-1,flatten_dim])
            layer_shape = self.get_layer_shape(self.Flatten)
            print('Flatten :',layer_shape)

            # FC1

            w4 = tf.Variable(tf.truncated_normal([layer_shape[1], self.D1_NODE], stddev=0.1),name="w4")
            b4 = tf.Variable(tf.constant(1.0, shape=[self.D1_NODE]),name = "b4")

            layer4_fccd = tf.matmul(self.Flatten, w4,name="D4") + b4
            self.layer4_actv = tf.nn.sigmoid(layer4_fccd,name = "D1_activation")

            layer_shape = self.get_layer_shape(self.layer4_actv)
            print('D1 :', layer_shape)
            #END

            # FC2
            w5 = tf.Variable(tf.truncated_normal([self.D1_NODE, self.D2_NODE], stddev=0.1),name="W3")
            b5 = tf.Variable(tf.constant(1.0, shape=[self.D2_NODE]),name = "b3")

            self.layer5_fccd = tf.matmul(self.layer4_actv, w5,name="D2") + b5


            self.layer5_out = tf.nn.softmax(self.layer5_fccd, name = "D2_activation")
            layer_shape = self.get_layer_shape(self.layer5_out)
            print('D2 :', layer_shape)
            #END

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.layer5_fccd))

            self.optimizer = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

            true = tf.arg_max(self.y, 1)
            predicted = tf.arg_max(self.layer5_out, 1)

            corrects = tf.cast(tf.equal(true, predicted), tf.float32)
            self.accuracy = tf.reduce_mean(corrects)

            self.init = tf.global_variables_initializer()
            self.saver=tf.train.Saver()

            print('Graph Structure Ready.')

    def fit(self, train_x, train_y, epochs, batchsize):
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init])

            nbtrain = len(train_x)
            nbbatches = int(np.ceil(nbtrain / float(batchsize)))
            print('no : ',nbbatches)

            for ep in range(epochs):

                start = 0
                loss = 0
                acc = 0
                for b in range(nbbatches):
                    end = min(start + batchsize,nbtrain-1)
                    batchx = train_x[start:end]
                    batchy = train_y[start:end]
                    #print("batch X shape ",batchx.shape," batch Y shape ",batchy.shape)
                    dict = {self.x: batchx, self.y: batchy}
                    _, bl, ba, lo = sess.run([self.optimizer, self.loss, self.accuracy, self.layer5_out],
                                             feed_dict=dict)
                    print("\rReading batch %d"%b,end="")

                    loss += bl
                    acc += ba
                    #print(lo)

                loss = loss / nbbatches
                acc = acc / nbbatches
                print("loss : %f, acc %f"%(loss,acc))
                self.saver.save(sess,"Weights/last")


obj = CNN(CONV=[8,16,32], POOL=[4,4,4], DENSE = [20,10])
obj.CNN_computation()
X_Train, Y_Train = load_MNIST('/home/sudip/tensorflow/train.csv')
obj.fit(X_Train, Y_Train, 10, 128)