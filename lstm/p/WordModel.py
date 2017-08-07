from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from ReadCorpus import *

def makenetwork(maxlen,vs):
    nw=Sequential()
    nw.add(LSTM(64,return_sequences=True,input_shape=(maxlen,vs)))
    nw.add(LSTM(128, return_sequences=False))
    nw.add(Dense(vs,activation="softmax"))
    nw.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=['accuracy'])
    nw.summary()
    return nw

def trainnetwork(nw,x,y,epochs,vs):
    for i in range(epochs):
        nw.fit(x,y,epochs=1,verbose=1)
        nw.save("Socialism")
        #testnetwork(sentence,nw,vs)
        generate(nw,"socialism is a",20,vs)

def testnetwork(sentence,nw,vs):
    words=sentence.split()
    total=len(sentence)
    wordvects=[]
    for w in words:
        vec=englishword2vec(vs,w)
        wordvects.append(vec)
    wordvects=np.asarray([wordvects])

    predicts=nw.predict(wordvects)
    #print("Predicted ",predicts.shape)
    predicts=np.reshape(predicts,(vs))

    p_word=englishvec2word(vs,predicts,5)
    #print(p_word)
    return p_word

def generate(nw,start,length,vs):
    sentence=start
    for w in range(length):
        next=testnetwork(start,nw,vs)
        sentence=sentence+" "+next[0]
        start=next[0]
    print(sentence)

vs=readenglishtext("got.txt")
x,y=loadenglishwords("got.txt",vs,3)
maxlen=len(x[0])
vs=len(x[0][0])
x=np.asarray(x)
y=np.asarray(y)
y=np.reshape(y,(len(y),vs))
print("X=",x.shape," Y=",y.shape)
nw=makenetwork(maxlen,vs)
trainnetwork(nw,x,y,1000,vs)
