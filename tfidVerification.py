#Author: Qianyu Deng
from sklearn.feature_extraction.text import TfidfTransformer

import os
import sys
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def getFilelist(argv) :
	path = argv[1]
	filelist = []
	files = os.listdir(path)
	for f in files :
		if(f[0] == '.') :
			pass
		else :
			filelist.append(f)
	return filelist,path


def Tfidf(filelist):
    corpus = []
    order = []
    for ff in filelist:
        order.append(ff[6:7])
        fname = path + ff
        f = open(fname, 'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    sFilePath = './tfidffile'
    if not os.path.exists(sFilePath):
        os.mkdir(sFilePath)


    for i in range(len(weight)):
        print u"--------Writing all the tf-idf in the", order[i], u" file into ", sFilePath + '/' + string.zfill(order[i], 5) + '.txt', "--------"
        f = open(sFilePath + '/' + string.zfill(order[i], 5) + '.txt', 'w+')
        for j in range(len(word)):
            f.write("["+order[i]+","+word[j]+"]" + "	" + str(weight[i][j]) + "\n")
        f.close()

if __name__ == "__main__":
    (allfile, path) = getFilelist(sys.argv)

    Tfidf(allfile)