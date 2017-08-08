import numpy as np
import csv
def readcorpustext(filename,writefile):
    f=open(filename,'r')
    wf=open(writefile,'a')
    line=f.readline()
    l=1
    while(line):
        print("Reading Line ",l)
        line=f.readline()
        wf.write(line.encode("unicode_escape").decode())
        l=l+1
        #wf.write("#-")
    wf.close()

def readunicodefromfile(filename,outfile):
    f=open(filename,'r')
    wf=open(outfile,'w')
    line=f.readline()
    while line:
        words=line.split(" ")
        for w in words:#for all words in line
            print(w)
            chars=w.split("\\")
            print("Chars-- ",chars)
            if(len(chars)>1):#valid word
                word=w.replace("\\"," \\")
                uniword=""
                for c in chars:# for all characters in the word
                    #print("\t",c)
                    if(len(c)>1) and c!="," and c!="(" and c!=")":#valid character
                        unichar=b'\\'+c.encode()
                        print("\t",c," Unicode ",unichar.decode("unicode_escape"))
                        uniword=uniword+unichar.decode("unicode_escape")
                wf.write(uniword+"|"+word+"\n")
        wf.write("\n")
        line=f.readline()
    wf.close()
    f.close()

def finddistinct(filename,wordfile,charfile):
    f=open(filename,'r')
    wf=open(wordfile,'w')
    cf=open(charfile,'w')
    reader=csv.reader(f,delimiter="|")
    words=[]
    allchars = []
    for row in reader:
        print(row)
        if(len(row)==2):
            words.append(row[0])
            for c in row[0]:
                #print(c.encode("unicode_escape").decode("utf-8"))
                allchars.append(c.encode("unicode_escape").decode("utf-8"))

    wordset=list(set(words))
    charset=list(set(allchars))
    print(charset)
    for w in range(len(wordset)):
        wf.write(wordset[w]+"\n")
    wf.close()
    for c in range(len(charset)):
        cf.write(charset[c]+"\n")

def loadchars(charfile):
    f=open(charfile,'r')
    line=f.readline()
    allchars=[]
    while line:
        allchars.append(line)

    total=len(allchars)
    return allchars,total

def getcharfromword(word):
    chars=[]
    for c in word:
        chars.append(c.encode("unicode_escape").decode("utf-8"))
    return chars,len(chars)

def loadwordsmultifile(wordfiles):
    allwords=[]
    charset=[]
    maxlen=0
    for w in wordfiles:
        aw,cs,ml=loadwords(w)
        allwords.extend(aw)
        charset.extend(cs)
        if(ml>maxlen):
            maxlen=ml
    charset=sorted(list(set(charset)))
    return allwords,charset,maxlen

def loadwords(wordfile):
    f=open(wordfile,'r')
    line=f.readline()
    allwords=[]
    allchars=[]
    maxlen=0
    while line:
        chars,wordlen=getcharfromword(line)
        if(wordlen>maxlen):
            maxlen=wordlen
            longestword=line
        for c in chars:
            allchars.append(c)
            allwords.extend(chars)
        line=f.readline()
    charset=sorted(list(set(allchars)))
    return allwords,charset,maxlen

def readenglishtext(filename):
    allwords=[]
    f=open(filename,'r')
    line=f.readline()
    while line:
        words=line.split()
        for w in words:
            w=w.strip(",").strip(".").strip()
            allwords.append(w)
        line=f.readline()
    wordset=list(set(allwords))
    f.close()

    f=open("EnglishVocabulary.txt","w")
    c=0
    for w in wordset:
        f.write(str(c)+" "+w+"\n")
        c=c+1
    f.close()
    vocabulary_size=len(wordset)
    return vocabulary_size

def englishword2vec(vs,word):
    f=open("EnglishVocabulary.txt","r")
    line=f.readline()
    vec=np.zeros(vs)
    while line:
        wi=line.split()
        if(wi[1].strip("\n")==word):
            ind=int(wi[0])
            vec[ind]=1
            break
        line=f.readline()
    f.close()
    return vec

def findtopinds(nb,predicts):#Finds the first Nb values and indices
    s_v=sorted(predicts,reverse=True)
    #print(predicts[0])
    tops=[]
    vals=list(predicts)
    for n in range(nb):
        tops.append(vals.index(s_v[n]))
    return tops

def englishvec2word(vs,vec,tops):
    topinds=findtopinds(tops,vec)
    #ind=np.argmax(vec)
    words=[]
    for t in topinds:
        f = open("EnglishVocabulary.txt", "r")
        line = f.readline()
        while line:
            wi = line.split()
            i = int(wi[0])
            if(i==t):
                word=wi[1].strip("\n")
                break
            line = f.readline()
        f.close()
        words.append(word)
    return words

def loadenglishwords(filename,vs,maxlen):
    text=[]
    f = open(filename, 'r')
    line = f.readline()
    while line:
        words = line.split()
        totalwords=len(words)
        i=0
        while(i<totalwords-1):
            w = words[i].strip(",").strip(".").strip()
            text.append(w)
            i=i+1
        line=f.readline()
    totalwords=len(text)
    print("Text Corpus loaded ",totalwords)
    x=[]
    y=[]
    w=0
    while(w<totalwords-maxlen):
        start=w
        end=start+maxlen
        print("Reading from ",start," to ",end)
        xs=[]
        ys=[]
        i = start
        while(i<end):
            wv=englishword2vec(vs,text[i])
            xs.append(wv)
            i=i+1
        nwv=englishword2vec(vs,text[i])
        ys.append(nwv)
        x.append(xs)
        y.append(ys)
        w=w+maxlen
    return x,y

#readcorpustext("part-00007_3","Corpus_unicode.txt")
#readunicodechars("Corpus_part-00000_unicode.txt","temp.txt")
#readunicodefromfile("Corpus_unicode.txt","All.txt")
#finddistinct("All.txt","allwords.txt","allchars.txt")
#a,b,c=loadwords("allwords.txt")
#print(b,c)

#vs=readenglishtext("socialism.txt")
#x,y=loadenglishwords("socialism.txt",vs)
