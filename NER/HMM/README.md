# HMM

We use HMM to find out-of-vocabulary (OOV) words when the training set and vocabulary size is relatively small. 

## Data

Prepare the train/test data in the following manner:
`B` indicates the beginning of a named entity, `I` indicates the body of a named entity, `O` indicates other words/characters.

```
美	B-LOC
国	E-LOC
的	O
华	B-PER
莱	I-PER
士	E-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

## Run

First, install all requirements:

```
pip3 install -r requirement.txt
```

Run the following command:

```
python3 main.py
```

The model will output scores of precision, recall, F1-score, and a confusion matrix. The model will be saved in `./ckpts/`.

To evaluate, run:

```
python3 test.py
```

## 隐马尔可夫模型（Hidden Markov Model，HMM）	 

隐马尔可夫模型描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程（李航 统计学习方法）。隐马尔可夫模型由初始状态分布，状态转移概率矩阵以及观测概率矩阵所确定。

命名实体识别本质上可以看成是一种序列标注问题，在使用HMM解决命名实体识别这种序列标注问题的时候，我们所能观测到的是字组成的序列（观测序列），观测不到的是每个字对应的标注（状态序列）。

**初始状态分布**就是每一个标注的初始化概率，**状态转移概率矩阵**就是由某一个标注转移到下一个标注的概率（就是若前一个词的标注为$tag_i$ ，则下一个词的标注为$tag_j$的概率为 $M_{ij}$），**观测概率矩阵**就是指在

某个标注下，生成某个词的概率。

HMM模型的训练过程对应隐马尔可夫模型的学习问题（李航 统计学习方法），

实际上就是根据训练数据根据最大似然的方法估计模型的三个要素，即上文提到的初始状态分布、状态转移概率矩阵以及观测概率矩阵，模型训练完毕之后，利用模型进行解码，即对给定观测序列，求它对应的状态序列，这里就是对给定的句子，求句子中的每个字对应的标注，针对这个解码问题，我们使用的是维特比（viterbi）算法。

具体的细节可以查看 `models/hmm.py`文件。

## Source
https://github.com/luopeixiang/named_entity_recognition









