import os
import h5py
import xlrd
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

class DrugClassifier():

    def __init__(self):
        self.threashold = 0.5
        self.words_list = []
        self.train_pVec = []


    def train(self, vocab_file):
        with open(vocab_file,'r',encoding = 'utf-8') as f: 
            vocab = f.read().splitlines()
        std_word_dict = self.createWordDict(vocab)
        sort_dict = sorted(std_word_dict.items(),key=lambda x:x[1],reverse=True)
        self.words_list = [i for i,_ in sort_dict]

        pVect = np.zeros([len(vocab),len(self.words_list)])
        pDenom = np.zeros([len(vocab),1])
        for i, words_i in tqdm(enumerate(vocab)):
            c = i
            word_vec = self.bagOfWords2VecMN(words_i)
            pVect[c] += word_vec
            pDenom[c] += np.sum(word_vec)
        self.train_pVec = np.zeros_like(pVect)
        for i in range(len(pVect)):
            if pDenom[i] == 0:
                pVect[i] = 0
                continue
            self.train_pVec[i] = pVect[i]/pDenom[i]

    def createWordDict(self, dataList):
        dataDict = {}
        for item in dataList:
            for character in item:
                if character in ['\u3000']:
                    continue
                if character not in dataDict.keys():
                    dataDict[character] = 1
                else:
                    dataDict[character] +=1
        return dataDict

    def is_drug(self, item):
        index_vec = self.index_in_wordList(item)
        if len(index_vec) == 0:
            return False
        p1 = np.sum(self.train_pVec[:,index_vec],1)
        if p1.max() > self.threashold:
            return True
        else:
            return False

    def bagOfWords2VecMN(self, inputSet):
        returnVec = np.zeros([len(self.words_list)])
        for word in inputSet:
            try:
                returnVec[self.words_list.index(word)] += 1
            except:
                continue
        return returnVec

    def index_in_wordList(self, inputWord):
        indexVec = []
        for word in inputWord:
            try:
                indexVec.append(self.words_list.index(word))
            except:
                continue
        return np.unique(indexVec)

def load(file_path):
    with open(file_path, 'rb') as f:
        c = pickle.load(f)
    return c

def save(c, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(c, f)




