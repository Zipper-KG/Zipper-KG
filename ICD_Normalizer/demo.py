from icd_normalizer import FindStdWord
import numpy as np
import pandas as pd
if __name__ == '__main__':
	disease_names = list(np.load('./data/disease_names.npy'))
	normalizer = FindStdWord(disease_names)
	test_path = './data/test_Disease_Data.csv'
	test_frame = pd.read_csv(test_path, encoding = 'gbk', header = None)
	test_data = []
	for i in test_frame:
	    test_data += list(test_frame[i])
	test_data = set(test_data)
	for i in test_data:
	    if type(i) == float:
	        continue
	    print(f'disease: {i}, normalized disease name: {normalizer(i)}')