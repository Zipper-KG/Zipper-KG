import jieba
import numpy as np
class PrePareRegionData:
    def __init__(self, regionWords):
        self.regionWords = regionWords
        self.wordList = self.createWordList()
        self.word2Vec = Word2Vec(self.wordList)
    def __call__(self):
        self.regionWordsMatrix = self.regionWords2Vec()
        return self.wordList, self.regionWordsMatrix
    def createWordList(self):
        wordList = self.createWordList_ch() + self.createWordList_jieba()
        return list(set(wordList))
    def createWordList_ch(self):
        wordList = []
        for item in self.regionWords:
            for character in item:
                if character not in wordList:
                    wordList.append(character)
                else:
                    continue
        return wordList
    def createWordList_jieba(self):
        wordList = []
        for item in self.regionWords:
            units = jieba.lcut(item)
            for character in units:
                if character not in wordList:
                    wordList.append(character)
                else:
                    continue
        return wordList
    def regionWords2Vec(self):
        regionMatrix = np.zeros([len(self.regionWords), len(self.wordList)])
        for i, word in enumerate(self.regionWords):
            regionMatrix[i] = self.word2Vec(word)
        return regionMatrix
class Word2Vec:
    def __init__(self, wordList):
        self.wordList = wordList
    def __call__(self, word):
        returnVec = np.zeros([len(self.wordList)])
        for i, unit in enumerate(self.wordList):
            if unit in word:
                returnVec[i] = 1.
        return returnVec
class Word2Index:
    def __init__(self, wordList):
        self.wordList = wordList
    def __call__(self, word):
        index = []
        for i, unit in enumerate(self.wordList):
            if unit in word:
                index.append(i)
        return index
class FindStdWord:
    def __init__(self, regionWords):
        initializer = PrePareRegionData(regionWords)
        self.regionWords = regionWords
        self.stopWords = self.__loadStopWords()
        self.wordList, self.regionWordsMatrix = initializer()
        self.word2Index = Word2Index(self.wordList)
        self.regionWordsLen = np.sum(self.regionWordsMatrix, 1)
        self.diseaseShort = list(np.load('./data/diseaseShortNameDict.npy'))
    def __loadStopWords(self):
    	stopWords = list(np.load('./data/chineseStopWords.npy'))
    	for disease_name in self.regionWords:
    		for i in stopWords:
    			if i in disease_name:
    				stopWords.remove(i)
    	return stopWords
    def __removeStopWord(self, text):
    	for i in self.stopWords:
    		if i in text:
    			text = ''.join(text.split(i))
    	return text
    def __replaceShortName(self, text):
    	for i in self.diseaseShort:
    		if i[0] in text:
    			text = i[1].join(text.split(i[0]))
    	return text
    def __preprocess(self, text):
    	text = self.__removeStopWord(text)
    	text = self.__replaceShortName(text)
    	return text
    def __findStd(self, text):
        index = self.word2Index(text)
        matchLen = np.sum(self.regionWordsMatrix[:,index],1)
        # matchEvaluate = np.min([matchLen/len(index),matchLen/self.regionWordsLen],0)
        matchEvaluate = matchLen/len(index) + matchLen/self.regionWordsLen
        max_i = np.argmax(matchEvaluate)
        if matchEvaluate[max_i] > 0.:
            return self.regionWords[max_i]
        else:
            return []
    def __call__(self, text):
        answers = []
        text = self.__preprocess(text)
        return self.__findStd(text)