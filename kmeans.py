#Author: Qian Wang
from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt
import numpy as np
import sys
import pyspark
from pyspark import SparkFiles
import os


sc = pyspark.SparkContext()


def k_means(filename = "gs://lisa_adv_data/tf_idf_matrix_big.txt", k = 10, top = 15, maxIter = 100, initialization = "random"):
    dataset = sc.textFile(filename)
    parsedData = dataset.map(lambda line: np.array([float(x) for x in line.split(' ')][1:]))

    clusters = KMeans.train(parsedData, k, maxIterations= maxIter , initializationMode=initialization)
    mapped_value = parsedData.map(clusters.predict).zip(parsedData)
    cluster_value = mapped_value.reduceByKey(lambda a,b: a + b)
    
    topics = []
    for i in cluster_value.collect():
        cluster_id = i[0]
        tf_idf_sum = sc.parallelize(list(i[1])).zipWithIndex()
        topics.append(
            [cluster_id, tf_idf_sum.takeOrdered(top, key=lambda x: -(x[0]))])
        
    words_index = []
    for each_cluster in topics:
        each = each_cluster[1]
        cluster_id = each_cluster[0]
        words = sc.parallelize(each).map(lambda x: x[1])
        words_index.append((cluster_id, words.collect()))

    return words_index


    


if __name__ == '__main__':
    
    #word_map = np.loadtxt("gs://lisa_adv_data/corpus.txt", dtype = 'string', delimiter=" ")
    words_index = k_means()
    
    path = os.path.join("gs://lisa_adv_data/", "topwords_big.txt")
    sc.parallelize(words_index).saveAsTextFile(path)
    
	