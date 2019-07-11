import codecs
import re
from nltk.tokenize import word_tokenize
import nltk

class Triple:
    def __init__(self, triple_line):
        self.left_term, self.right_term, *self.relations = re.split('\t', triple_line.strip())
        self.tok_left_term = word_tokenize(self.left_term)
        self.tok_right_term = word_tokenize(self.right_term)
        self.offset_vector = None

    def set_offset_vector(self, offset_vector):
        self.offset_vector = offset_vector

    def to_string(self):
        out_str = self.left_term+"\t"+self.right_term+"\t"+self.relation
        if self.offset_vector is not None:
            out_str += "\t" + str(self.offset_vector)
        return out_str

class DataReader:
    def read_data(self, filename):
        triple_list = []
        labels = []
        with codecs.open(filename, 'r', 'utf-8') as file:
            for line in file:
                triple = Triple(line)
                triple_list.append(triple)
                labels.append(triple.relations)
        return triple_list, labels