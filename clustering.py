#Author: Qianyu Deng
import time
import os
import codecs
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def merge_file():
    path = "review_big/"
    resName = "review_big/review_clean_big.txt"
    if os.path.exists(resName):
        os.remove(resName)
    result = codecs.open(resName, 'w', 'utf-8')

    num = 0
    while num <= 10:
        name = "review_clean_big"+str(num)
        fileName = path + str(name) + ".txt"
        source = open(fileName, 'r')
        line = source.readline()
        while line != "":
            line = unicode(line, "utf-8")
            result.write(line)
            line = source.readline()
        else:
            print 'End file: ' + str(num)
            result.write('\r\n')
            source.close()
        num = num + 1

    else:
        print 'End All'
        result.close()

def tfid():
    corpus = []

    for line in open('/home/qydeng/Desktop/Prediction_model/Prediction_model/review_clean_big.txt', 'r').readlines():
        #print line
        corpus.append(line.strip())
        # print corpus
    time.sleep(5)

    vectorizer = CountVectorizer()

    transformer = TfidfTransformer()

    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()

    weight = tfidf.toarray()
    #resName = "/home/qydeng/Desktop/Prediction_model/Prediction_model/review_big_tfidf.txt"
    #result = codecs.open(resName, 'w', 'utf-8')
    #for i in range(len(weight)):
    #    for j in range(len(word)):
    #        result.write("["+str(i)+","+word[j]+"]"+","+str(weight[i][j])+"\n")

    #resName = "/home/qydeng/Desktop/Prediction_model/Prediction_model/review_big_tfidf.txt"
    #result = codecs.open(resName, 'w', 'utf-8')
    #for j in range(len(word)):
    #    result.write(word[j] + ' ')
    #result.write('\n')

    #for i in range(len(weight)):
    #    print "output file "+str(i)
    #    for j in range(len(word)):
    #        result.write(str(weight[i][j]) + ' ')
    #    result.write('\n')
    #result.close()
    return word,weight
def listSum(a,b):
    res = []
    for i in range(len(a)):
        sum = a[i]+b[i]
        res.append(sum)
    return res

if __name__ == '__main__':
    weight = []
    inputName = "/home/qydeng/Desktop/Prediction_model/Prediction_model/tf_idf_matrix_big.txt"
    f = open(inputName,"r+")
    line = f.readline()
    while(line!=""):
        list = line.strip('\n').split(" ")
        list = map(float,list[1:])
        weight.append(list)
        line = f.readline()

    #merge_file()
    #weight = tfid()

    clf = KMeans(n_clusters=10)
    s = clf.fit(weight)
    print s

    print(clf.cluster_centers_)

    print(clf.labels_)
    i = 0
    label = codecs.open("/home/qydeng/Desktop/Prediction_model/Prediction_model/QianyuResult/kmeans_label.txt", 'w', 'utf-8')
    while i <= len(clf.labels_)-1:
        label.write(str(i)+" "+str(clf.labels_[i])+"\n")
        i = i + 1
    print(clf.inertia_)
    groupWeight = []
    for i in range(10):
        groupWeight.append([0]*len(weight[0]))
    for i in range(len(clf.labels_)):
        groupWeight[clf.labels_[i]] = listSum(groupWeight[clf.labels_[i]],weight[i])
    groupWeightSort = []
    for i in range(len(groupWeight)):
        sort = [j[0] for j in sorted(enumerate(groupWeight[i]), reverse=True,key=lambda x:x[1])]
        groupWeightSort.insert(i,sort[0:15])
    for i in range(len(groupWeightSort)):
        print groupWeightSort[i]

    wordFileName = "/home/qydeng/Desktop/Prediction_model/Prediction_model/corpus_big.txt"
    topWords = codecs.open("/home/qydeng/Desktop/Prediction_model/Prediction_model/QianyuResult/topWordsSklearn.txt",'w', 'utf-8')
    wordFile = open(wordFileName,"r+")
    word = []
    wordline = wordFile.readline()
    word = wordline.split(" ")
    for i in range(len(groupWeightSort)):
        topWords.write("[")
        for j in range(len(groupWeightSort[i])):
            topWords.write(word[groupWeightSort[i][j]]+" ")
        topWords.write("]\n")



