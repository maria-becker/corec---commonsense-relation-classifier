print("Starting all")
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from data_reader_mlc import DataReader

import os

from config import Config
import sys
print("First Imports finished")

from sklearn.model_selection import train_test_split
class VectorProcessor:
    def read_vectors(self, vector_file, vocab):
        try:
            self.word_vectors = KeyedVectors.load_word2vec_format(vector_file, binary=True)
        except ValueError:
            self.word_vectors = KeyedVectors.load_word2vec_format(vector_file, binary=False)
        self.random = np.random.rand(len(self.word_vectors.syn0[0]))
        self.word2idx = {x:i for i,x in enumerate(["<UNK>"]+list(x for x in self.word_vectors.vocab.keys() if x in vocab))}
        self.idx2word = {i:x for x,i in self.word2idx.items()}
        print(len(self.word2idx))
        self.UNK = np.random.normal(size=(len(self.word_vectors.syn0[1]),))
        print(self.UNK.shape)
    def get_word_vector(self, word):
        if word in self.word_vectors.vocab:
            #print((self.word_vectors[word].shape))
            return self.word_vectors[word]
        else:
            return self.UNK
    def get_vocab(self):
        return list(self.word2idx.keys())
    def get_matrix(self):
        return np.array([self.UNK]+[self.get_word_vector(y) for x,y in sorted(self.idx2word.items())[1:]])
    def get_word_ids(self, words):
 
        liste = []
        for word in words:
            if word in self.word2idx:
                liste.append(self.word2idx[word])
            elif word.lower() in self.word2idx:
                liste.append(self.word2idx[word.lower()])
            else:
                liste.append(self.word2idx["<UNK>"])
 
        return liste
    def get_average_word_vector(self, word_list):
        embedding_size = len(self.word_vectors.syn0[0])
        out_vector = np.zeros(embedding_size)
        division_int = len(word_list)

        for word in word_list:
            vector = self.get_word_vector(word)
            if vector is not None:
                out_vector = np.add(out_vector, np.array(vector))
            else:
                division_int -= 1

        if division_int == 0:
            return out_vector
        else:
            return out_vector/division_int

    def get_offset_average_vector(self, word_list1, word_list2):
        vector1 = self.get_average_word_vector(word_list1)
        vector2 = self.get_average_word_vector(word_list2)

        return np.subtract(vector1, vector2)

    def get_concat_average_vector(self, word_list1, word_list2):
        vector1 = self.get_average_word_vector(word_list1)
        vector2 = self.get_average_word_vector(word_list2)
        return np.hstack((vector1, vector2))


class VectorOffsetClassification:
    def __init__(self, classification_file, vector_file, processing_folder='tmp'):
        print(classification_file, vector_file)
        if not os.path.exists(processing_folder):
            os.makedirs(processing_folder)

        reader = DataReader()
        self.classification_filename = os.path.basename(classification_file)
        self.processing_folder = processing_folder
        self.triple_list, self.labels = reader.read_data(classification_file)
        self.vector_processor = VectorProcessor()
        self.vector_processor.read_vectors(vector_file)
        self.offset_file = self.processing_folder+'/'+self.classification_filename+'.avg_off_vecs'

    def preprocess(self):
        offset_vectors = []

        for triple in self.triple_list:
            offset_vector = self.vector_processor.get_concat_average_vector(triple.tok_left_term, triple.tok_right_term)
            offset_vectors.append(offset_vector)

        np.savetxt(self.offset_file, offset_vectors, newline="\n")

    def load_vectors(self):
        self.offset_vectors = np.loadtxt(self.offset_file)

    def cross_validation(self):
        self.load_vectors()
        #log_reg = LogisticRegression()
        log_reg = LinearSVC()
        p_grid = {"C": [1, 10, 100]}

        x_train, x_test, y_train, y_test = train_test_split(self.offset_vectors, self.labels, test_size=0.1, random_state=0)

        clf = GridSearchCV(log_reg, p_grid, cv=10, scoring='f1_macro')
        clf.fit(x_train, y_train)

        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
print("Starting next imports")
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(sys.argv)

print("Processing Data")
reader = DataReader()
triple_list, labels = reader.read_data(sys.argv[1])
dev_list, dev_labels = reader.read_data(sys.argv[4])
test_list, test_labels = reader.read_data(sys.argv[5])
vocab = {x for triple in triple_list+dev_list+test_list for x in triple.tok_left_term+triple.tok_right_term}
vocab.add("<UNK>")
print("Processed Data")
label_set = set([y for x in labels+dev_labels+test_labels for y in x])
label2idx = {x:i for i,x in enumerate(label_set)}
idx2label = {i:x for x,i in label2idx.items()}

x_train = triple_list
y_train = labels
x_val = dev_list
y_val = dev_labels
x_test = test_list
y_test = test_labels
print("Processing Vectors")
vector_processor = VectorProcessor()
vector_processor.read_vectors(sys.argv[2], vocab)        
print("Processed Vectors")
matrix = vector_processor.get_matrix()

def validate(model, x_val, y_val):
    losses = []
    loss_function = nn.BCEWithLogitsLoss()
    for i,(x, y) in enumerate(zip(x_train, y_train)):
        left = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_left_term)), requires_grad=False).cuda()
        right = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_right_term)), requires_grad=False).cuda()
        
        labelvector = [0.0]*len(label_set)
        for label in y:
            labelvector[label2idx[label]]=1.0
        label = autograd.Variable(torch.FloatTensor(labelvector), requires_grad=False).cuda()
        log_probs = model(left, right).view(1, -1)
        loss = loss_function(log_probs, label.view(1, -1))

        losses.append(loss.cpu().data)
        del log_probs
        del loss
        del left
        del right
        del label
    print("val_loss", sum(losses))
    return sum(losses)

def early_stopping(liste, n):
    results=liste[-n:]
    if len(liste) < n:
        return False
    boollist=[]
    for i in range(1, len(results)):  
        boollist.append((results[0].numpy() < results[i].numpy()).item())
    print(boollist)
    return all(boollist)

def train(Predictor, x_train, y_train, x_val, y_val, x_test, y_test, matrix):
    model = Predictor(matrix.shape[0], matrix.shape[1], len(label_set), matrix).cuda()
    print("Made Model")
    val_losses = []
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([x for x in model.parameters() if x.requires_grad], lr=0.001) #Hyperparameter
    print("Starting training")
    for epoch in range(1000):
        total_loss = torch.Tensor([0])
        for i,(x, y) in enumerate(zip(x_train, y_train)):
            left = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_left_term)), requires_grad=False).cuda()
            right = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_right_term)), requires_grad=False).cuda()
            labelvector = [0.0]*len(label_set)
            for label in y:
                labelvector[label2idx[label]]=1.0
            label = autograd.Variable(torch.FloatTensor(labelvector), requires_grad=False).cuda()
            log_probs = model(left, right).view(1, -1)
            loss = loss_function(log_probs, label.view(1, -1))
            total_loss += loss.cpu().data
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            del log_probs
            del loss
            del left
            del right
            del label
        val_losses.append(validate(model, x_val, y_val))
        if early_stopping(val_losses, 5):
            break
        print("train_loss", epoch, total_loss)
        
    predictions = []
    print("\t"+"\t".join(label_set))
    for i,(x, y) in enumerate(zip(x_test, y_test)):
        left = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_left_term)), requires_grad=False).cuda()
        right = autograd.Variable(torch.LongTensor(vector_processor.get_word_ids(x.tok_right_term)), requires_grad=False).cuda()
        labelvector = [0.0]*len(label_set)
        for label in y:
            labelvector[label2idx[label]]=1.0
        predictions = model(left, right).cpu().data.numpy()
        try:
            pred_string = "\t".join([idx2label[i]+"|"+str(1/(1+math.exp(- res))) for i, res in enumerate(predictions.tolist()[0])])
        except OverflowError:
            print ("OverflowError")
            print (predictions)
            continue
        except Exception as e:
            print ("OtherException")
            print (predictions)
            continue
        gold_string = "\t".join([idx2label[i]+"|"+str(res) for i, res in enumerate(labelvector)])
        print(" ".join(x.tok_left_term), " ".join(x.tok_right_term), pred_string, "###", gold_string, "\n", sep="\t")
    


from importlib import import_module
Predictor = import_module(sys.argv[3]).Predictor 
print(sys.argv)
for i in range(1):
    train(Predictor, x_train, y_train, x_val, y_val, x_test, y_test, matrix)



