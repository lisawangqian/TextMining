#Author: Qian Wang
from mrjob.job import MRJob
import mrjob
from mrjob.step import MRStep
import os
import sys
from mrjob.protocol import RawProtocol
from mrjob.protocol import JSONProtocol
import collections
import itertools
from math import log10
from math import sqrt
import re

#delimiter = re.compile("\s")

class MR_tif_idf(MRJob):
    
    
    total_num_doc = 200      

    def tf_idf_mapper(self, _, line):
        terms =line.strip().split(" ")
       
        doc = int(terms[0])
        values = terms[1:]
        #print values
        word_counts = collections.Counter()
        for each in values:
            word_counts[each]+=1
        
        for each in word_counts:
            #yield each, (doc, 1+log10(word_counts[each]))
            yield each, (doc, word_counts[each])
        
        
    def tf_idf_reducer(self, term, values):
        vs = list(values)
        #idf = log10(float(self.total_num_doc)/len(vs))
        idf = log10((1+float(self.total_num_doc))/(1+len(vs)))
        for doc, tf in vs:
            yield doc, (term, tf*idf)
        
        
    def normalizing_reducer(self, doc, values):
        v1, v2 = itertools.tee(values)
        length = sqrt(sum([tf*tf for _, tf in v1]))
        for term, tf in v2:
            yield (doc, term), tf/length 
        
    def steps(self):
        return [MRStep(mapper = self.tf_idf_mapper, reducer  = self.tf_idf_reducer),
                MRStep(reducer = self.normalizing_reducer)]
        


if __name__ == '__main__':
    MR_tif_idf.run()