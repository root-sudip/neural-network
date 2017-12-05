import tensorflow as tf
import numpy as np


from PIL import Image

import matplotlib
import cv2 as cv

from data_representation import load_MNIST, load_MNIST_test,load_data




class CNN:
    def __init__(self, CONV, POOL,  DENSE):
        #pass
        self.CONV = CONV
        self.POOL = POOL
        self.D2_NODE = DENSE

        self.INPUT_SHAPE = [None, 28, 28, 3]
        self.OUTPUT_SHAPE = [None, 10]

        self.CONV1_F = [5, 5, 3, CONV[0]] #
        self.CONV1_S = [1, 1, 1, 1]
        self.CONV1_POOL = [1, POOL[0], POOL[0], 1]
        self.CONV1_POOL_S = [1, 1, 1, 1]

        self.CONV2_F = [5, 5, CONV[0], CONV[1]]
        self.CONV2_S = [1, 1, 1, 1]
        self.CONV2_POOL = [1,POOL[1], POOL[1], 1]
        self.CONV2_POOL_S = [1, 2, 2, 1]

        self.CONV3_F = [5, 5, CONV[1],CONV[2]]
        self.CONV3_S = [1, 1, 1, 1]
        self.CONV3_POOL = [1, POOL[2], POOL[2],1]
        self.CONV3_POOL_S = [1, 2, 2, 1]

        ####################

                    #deconvolution

        ###################

        self.CONV4_F = [5, 5, CONV[2], CONV[3]]
        self.CONV4_S = [1, 1, 1, 1]
        self.CONV4_POOL = [1, POOL[3], POOL[3], 1]
        self.CONV4_POOL_S = [1, 1, 1, 1]

        self.CONV5_F = [5, 5, CONV[3], CONV[4]]
        self.CONV5_S = [1, 1, 1, 1]
        self.CONV5_POOL = [1, POOL[3], POOL[3], 1]
        self.CONV5_POOL_S = [1, 1, 1, 1]

        self.CONV6_F = [5, 5, CONV[4], CONV[5]]
        self.CONV6_S = [1, 1, 1, 1]
        self.CONV6_POOL = [1, POOL[3], POOL[3], 1]
        self.CONV6_POOL_S = [1, 1, 1, 1]

        self.CONV7_F = [5, 5, CONV[5], CONV[6]]
        self.CONV7_S = [1, 1, 1, 1]
        self.CONV7_POOL = [1, POOL[3], POOL[3], 1]
        self.CONV7_POOL_S = [1, 1, 1, 1]

        #self.D1_NODE = DENSE[0]
        #self.D2_NODE = DENSE[1]

    def get_layer_shape(self,layer):
        sh=tf.Tensor.get_shape(layer)
        sh=[sh[j].value for j in range(len(sh))]
        return sh

    def flatten_array(self, array):
        shape = array.get_shape().as_list()
        return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

    def visualize(self, image = None ,num = None):
        pass
        #img = Image.fromarray(image,num)
        #img.save('out/'+str(num)+'.png')


    def CNN_computation(self):
        self.mygraph = tf.Graph()
        print('Computation graph created.')
        with self.mygraph.as_default():

            self.x = tf.placeholder(dtype = tf.float32, shape = self.INPUT_SHAPE, name = "X")  #network input shape
            self.y = tf.placeholder(dtype = tf.float32, shape = self.OUTPUT_SHAPE , name = "Y") #output shape

            # CONV1
            w1 = tf.Variable(tf.truncated_normal(self.CONV1_F, stddev=0.1),name = "W1", trainable=False)
            b1 = tf.Variable(tf.truncated_normal([self.CONV[0]], stddev=0.1),name="b1", trainable=False)

            layer1_conv = tf.nn.conv2d(self.x, w1, self.CONV1_S, padding = 'SAME',name = "C1_layer")

            layer_shape = self.get_layer_shape(layer1_conv)
            print('CONV1 : ', layer_shape)

            layer1_activaion = tf.nn.sigmoid(layer1_conv + b1,name = "l1_activation")
            self.layer1_pool = tf.nn.max_pool(layer1_activaion, self.CONV1_POOL, self.CONV1_POOL_S, padding = 'SAME',name = "l1_pool")
            layer_shape=self.get_layer_shape(self.layer1_pool)
            print('POOL1 : ',layer_shape)
            print("*****************************")
            #END

            # conv2
            w2 = tf.Variable(tf.truncated_normal( self.CONV2_F, stddev=0.1), name = "W2", trainable=False)
            b2 = tf.Variable(tf.truncated_normal([self.CONV[1]], stddev=0.1), name="b1", trainable=False)

            layer2_conv = tf.nn.conv2d(self.layer1_pool, w2, self.CONV2_S, padding = 'SAME', name = "C2_layer")

            layer_shape = self.get_layer_shape(layer2_conv)
            print('CONV2 : ', layer_shape)

            layer2_activaion = tf.nn.sigmoid(layer2_conv + b2, name = "l2_activation")
            self.layer2_pool = tf.nn.max_pool(layer2_activaion, self.CONV2_POOL, self.CONV2_POOL_S, padding = 'SAME',name = "l2_pool")
            layer_shape = self.get_layer_shape(self.layer2_pool)
            print('POOL2 : ', layer_shape)
            print("*****************************")
            #END

            # conv3
            w3 = tf.Variable(tf.truncated_normal(self.CONV3_F, stddev=0.1), name = "W3",  trainable=False)
            b3 = tf.Variable(tf.truncated_normal([self.CONV[2]], stddev=0.1), name = "b3",  trainable=False)

            layer3_conv = tf.nn.conv2d(self.layer2_pool, w3, self.CONV3_S, padding = 'SAME', name = "C3_layer")

            layer_shape = self.get_layer_shape(layer3_conv)
            print('CONV3 : ', layer_shape)

            layer3_activaion = tf.nn.sigmoid(layer3_conv + b3, name = "l3_activation")
            self.layer3_pool = tf.nn.max_pool(layer3_activaion, self.CONV3_POOL, self.CONV3_POOL_S, padding = 'SAME',name = "l3_pool")
            layer_shape = self.get_layer_shape(self.layer3_pool)
            print('POOL3 : ', layer_shape)
            print("*****************************")
            #END

            #Flatten
            flatten_dim = layer_shape[-1]*layer_shape[-2]*layer_shape[-3]

            self.Flatten=tf.reshape(self.layer3_pool,[-1,flatten_dim])
            layer_shape = self.get_layer_shape(self.Flatten)
            print('Flatten :',layer_shape)
            print("*****************************")

            
            #mlp ****************************************************************************
            #MLP layer1
            mlp_w1 = tf.Variable(tf.truncated_normal([layer_shape[1], 128], stddev=0.1), name="mlp_w1", trainable=True)
            mlp_b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="mlp_b1", trainable=True)

            mlp_mul1 = tf.matmul(self.Flatten, mlp_w1, name="Multiply1")
            self.mlp_h1 = tf.add(mlp_mul1, mlp_b1, name="addition1")
            self.layer1_out = tf.nn.tanh(self.mlp_h1, name="MLPLayer1Activation")

            #MLP layer2
            mlp_w2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1), name="mlp_w2", trainable=True)
            mlp_b2 = tf.Variable(tf.constant(0.1, shape=[256]), name="mlp_b2", trainable=True)

            mlp_mul2 = tf.matmul(self.layer1_out, mlp_w2, name="Multiply2")
            self.mlp_h2 = tf.add(mlp_mul2, mlp_b2, name="addition2")
            self.layer2_out = tf.nn.tanh(self.mlp_h2, name="MLPLayer2Activation")

            #MLP layer3
            mlp_w3 = tf.Variable(tf.truncated_normal([128, 2], stddev=0.1), name="mlp_w3", trainable=True)
            mlp_b3 = tf.Variable(tf.constant(0.1, shape=[2]), name="mlp_b3", trainable=True)

            mlp_mul3 = tf.matmul(self.layer2_out, mlp_w3, name="Multiply3")
            self.mlp_h3 = tf.add(mlp_mul3, mlp_b3, name="addition3")
            self.layer3_out = tf.nn.softmax(self.mlp_h3, name="Layer3Activation")

            #calculating the accuracy and loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.mlp_h3))

            self.optimizer = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

            true = tf.arg_max(self.y, 1)
            predicted = tf.arg_max(self.layer3_out, 1)

            corrects = tf.cast(tf.equal(true, predicted), tf.float32)
            self.accuracy = tf.reduce_mean(corrects)

            #end mlp ************************************************************************

            #reshaping
            self.reshaped = tf.reshape(self.Flatten, [-1, 7,7,32])
            layer_shape = self.get_layer_shape(self.reshaped)
            print('reshaped : ', layer_shape)
            print("*****************************")


            #DECONV1 ************************************************************************

            w4 = tf.Variable(tf.truncated_normal(self.CONV4_F, stddev=0.1), name="W4",  trainable=False)
            b4 = tf.Variable(tf.truncated_normal([self.CONV[3]], stddev=0.1), name="b4",  trainable=False)

            self.layer4_deconv = tf.nn.conv2d(self.reshaped, w4, self.CONV4_S, padding='SAME', name="DC1_layer")

            #self.layer4_deconv = tf.nn.conv2d_transpose(self.reshaped, 32, strides= (1,1), padding = 'SAME', name = "DC1_layer")

            layer_shape = self.get_layer_shape(self.layer4_deconv)
            print('DECONV1 : ', layer_shape)

            layer4_activaion = tf.nn.sigmoid(self.layer4_deconv + b4, name="l4_activation")

            us1=tf.image.resize_images(layer4_activaion,[14,14])
            layer_shape = self.get_layer_shape(us1)
            print('UPSAMPLE : ', layer_shape)
            print("*****************************")

            # DECONV2 ************************************************************************

            w5 = tf.Variable(tf.truncated_normal(self.CONV5_F, stddev=0.1), name="W5",  trainable=False)
            b5 = tf.Variable(tf.truncated_normal([self.CONV[4]], stddev=0.1), name="b5",  trainable=False)

            self.layer5_deconv = tf.nn.conv2d(us1, w5,self.CONV5_S, padding='SAME', name="DC2_layer")

            layer_shape = self.get_layer_shape(self.layer5_deconv)
            print('DECONV2 : ', layer_shape)

            layer5_activaion = tf.nn.sigmoid(self.layer5_deconv + b5, name="l5_activation")

            us2 = tf.image.resize_images(layer5_activaion, [20, 20])
            layer_shape = self.get_layer_shape(us2)
            print('UPSAMPLE : ', layer_shape)
            print("*****************************")

            # DECONV3 ************************************************************************

            w6 = tf.Variable(tf.truncated_normal(self.CONV6_F, stddev=0.1), name="W6",  trainable=False)
            b6 = tf.Variable(tf.truncated_normal([self.CONV[5]], stddev=0.1), name="b6",  trainable=False)

            self.layer6_deconv = tf.nn.conv2d(us2, w6, self.CONV6_S, padding='SAME', name="DC3_layer")


            layer_shape = self.get_layer_shape(self.layer6_deconv)
            print('DECONV3 : ', layer_shape)

            layer6_activaion = tf.nn.sigmoid(self.layer6_deconv + b6, name="l6_activation")

            us3 = tf.image.resize_images(layer6_activaion, [24, 24])
            layer_shape = self.get_layer_shape(us3)
            print('UPSAMPLE : ', layer_shape)
            print("*****************************")

            # DECONV4 ************************************************************************

            w7 = tf.Variable(tf.truncated_normal(self.CONV7_F, stddev=0.1), name="W7",  trainable=False)
            b7 = tf.Variable(tf.truncated_normal([self.CONV[6]], stddev=0.1), name="b7",  trainable=False)

            self.layer7_deconv = tf.nn.conv2d(us3, w7, self.CONV7_S, padding='SAME', name="DC4_layer")

            layer_shape = self.get_layer_shape(self.layer7_deconv)
            print('DECONV4 : ', layer_shape)

            layer7_activaion = tf.nn.sigmoid(self.layer7_deconv + b7, name="l6_activation")

            self.us3 = tf.image.resize_images(layer7_activaion, [28, 28])
            layer_shape = self.get_layer_shape(self.us3)
            print('UPSAMPLE : ', layer_shape)

            print("*****************************")

            ###**************************END LAYERS******************************

            self.loss = tf.reduce_mean(tf.squared_difference(self.x,self.us3,name="L2Loss"))

            self.optimizer = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

            #true = tf.arg_max(self.y, 1)
            #predicted = tf.arg_max(self.layer5_out, 1)

            #corrects = tf.cast(tf.equal(true, predicted), tf.float32)
            #self.accuracy = tf.reduce_mean(corrects)

            self.init = tf.global_variables_initializer()
            self.saver=tf.train.Saver()

            print('Graph Structure Ready.')

    def fit(self, train_x, train_y, epochs, batchsize):
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init]) #saver.restore(sess,savepath)

            nbtrain = len(train_x)
            nbbatches = int(np.ceil(nbtrain / float(batchsize)))
            print('no : ',nbbatches)

            for ep in range(epochs):

                print('Epoch : ',ep)

                start = 0
                loss = 0
                acc = 0
                for b in range(nbbatches):
                    end = min(start + batchsize,nbtrain-1)
                    batchx = train_x[start:end]
                    batchy = train_y[start:end]
                    #print("batch X shape ",batchx.shape," batch Y shape ",batchy.shape)
                    dict = {self.x: batchx}
                    op, lo = sess.run([self.optimizer, self.loss], feed_dict=dict)
                    print("\rReading batch %d"%b,end="")
                    print()

                    loss += lo

                    #print(lo)
                loss = loss / nbbatches
                #acc = acc / nbbatches
                print("loss : %f : "%(loss))
                self.saver.save(sess,"Weights/last")

    def predict(self, images, number_of_images, visualize=None):
        loss = 0
        with tf.Session(graph=self.mygraph) as sess:
            sess.run([self.init])
            new_saver = tf.train.import_meta_graph('Weights/last.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('Weights/'))

            for i in range(number_of_images):
                print("\rImage Number %d" % i, end="")
                reshape = np.reshape(images[i, :, :, :],(1,28,28,1))
                dict = {self.x: reshape}
                op, lo, out_last = sess.run([self.optimizer, self.loss, self.us3], feed_dict=dict)
                if visualize == True:
                    self.visualize(out_last,reshape, i)
                loss += lo

            total_loss = loss / number_of_images
            print('Loss : ', total_loss)

    def visualize(self, image, given, number):
        reshape = np.reshape(image, (28, 28)) * 255
        reshape = reshape.astype(int)

        reshape_given = np.reshape(given, (28, 28)) * 255
        reshape_given = reshape.astype(int)

        #print(reshape.shape)
        im = Image.fromarray(reshape.astype('uint8'))
        im.save('out/' + str(number) + '.png')

        im = Image.fromarray(reshape_given.astype('uint8'))
        im.save('given/' + str(number) + '.png')


obj = CNN(CONV=[8,16,32,32,16,8,3], POOL=[4,4,4,4],DENSE = [20,10])
obj.CNN_computation()


input = input('Training(Train)/Testing(Test)?')


if input == 'Train':
    X_Train, Y_Train = load_data(filename='label.csv', samples=287961)
    X_Train = X_Train/255 #normalizing
    obj.fit(X_Train, Y_Train, epochs=40, batchsize=128)

elif input == 'Test':
    print('passing')
    #X_Test, Y_test = load_MNIST_test('/home/sudip/tensorflow/test.csv')
    #X_Test = X_Test/255 #normalizing
    #obj.predict(X_Test,28000,True)

else:
    print('You should write Train/Test.')
