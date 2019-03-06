# -*- coding: utf-8 -*-
"""
预处理
corpus其实已经非常好了，已经整理成[label, text]的形式
我们需要做的就是，将text分词,去停用词
"""
from tools.tokenizer.wordCut import WordCut


# word_divider = WordCut()
# file_path =r'..\data\cnews\cnews.val.txt'
# write_path =r'..\data\cnews\val_token.txt'
# with open(write_path, 'w',encoding='utf-8') as w:
#     with open(file_path, 'r',encoding='utf-8') as f:
#         for line in f.readlines():
#             line = line.strip()
#             token_sen = word_divider.seg_sentence(line.split('\t')[1])
#             w.write(line.split('\t')[0] + '\t' + token_sen + '\n')


"""
val数据集最短...所有调通它的分词后
我们接下来就要考虑，将train和test文件也分词处理
保存各自的token文件到本地

多进程处理
"""
import multiprocessing


tmp_catalog = '../data/cnews/'
file_list = [tmp_catalog+'cnews.train.txt', tmp_catalog+'cnews.test.txt']
write_list = [tmp_catalog+'train_token.txt', tmp_catalog+'test_token.txt']

def tokenFile(file_path, write_path):
    word_divider = WordCut()
    with open(write_path, 'w', encoding='utf-8') as w:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                token_sen = word_divider.seg_sentence(line.split('\t')[1])
                w.write(line.split('\t')[0] + '\t' + token_sen + '\n')
    print(file_path + ' has been token and token_file_name is ' + write_path)

pool = multiprocessing.Pool(processes=4)
for file_path, write_path in zip(file_list, write_list):
    pool.apply_async(tokenFile, (file_path, write_path, ))
pool.close()
pool.join() # 调用join()之前必须先调用close()
print("Sub-process(es) done.")

"""
多进程速度还是不错的，但是跟木板效应一样，
最终的执行总时间，还是跟最大的文件有关。

"""

def constructDataset(path):
    """
    path: file path
    rtype: lable_list and corpus_list
    """
    label_list = []
    corpus_list = []
    with open(path, 'r', encoding='utf-8') as p:
        for line in p.readlines():
            label_list.append(line.split('\t')[0])
            corpus_list.append(line.split('\t')[1])
    return label_list, corpus_list

# tmp_catalog = '/home/zhouchengyu/haiNan/textClassifier/data/cnews/'
file_path = 'val_token.txt'
val_label, val_set = constructDataset(tmp_catalog+file_path)
print(len(val_set))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# tmp_catalog = '/home/zhouchengyu/haiNan/textClassifier/data/cnews/'
write_list = [tmp_catalog+'train_token.txt', tmp_catalog+'test_token.txt']

tarin_label, train_set = constructDataset(write_list[0]) # 50000
test_label, test_set = constructDataset(write_list[1]) # 10000
# 计算tf-idf
corpus_set = train_set + val_set + test_set # 全量计算tf-idf
print("length of corpus is: " + str(len(corpus_set)))
vectorizer = CountVectorizer(min_df=1e-5) # drop df < 1e-5,去低频词
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_set))
words = vectorizer.get_feature_names()
print("how many words: {0}".format(len(words)))
print("tf-idf shape: ({0},{1})".format(tfidf.shape[0], tfidf.shape[1]))

"""
上面有一个容易忽略的点：
计算tf-idf需要将train, val, test三方面的数据集全部计算，这样提取到的特征才更加准确
"""
from sklearn import preprocessing

# encode label
corpus_label = tarin_label + val_label + test_label
encoder = preprocessing.LabelEncoder()
corpus_encode_label = encoder.fit_transform(corpus_label)
train_label = corpus_encode_label[:50000]
val_label = corpus_encode_label[50000:55000]
test_label = corpus_encode_label[55000:]
# get tf-idf dataset
train_set = tfidf[:50000]
val_set = tfidf[50000:55000]
test_set = tfidf[55000:]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# LogisticRegression classiy model
lr_model = LogisticRegression()
lr_model.fit(train_set, train_label)
print("val mean accuracy: {0}".format(lr_model.score(val_set, val_label)))
y_pred = lr_model.predict(test_set)
print(classification_report(test_label, y_pred))

# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(n_estimators=200, random_state=1080)
rf_model.fit(train_set, train_label)
print("val mean accuracy: {0}".format(rf_model.score(val_set, val_label)))
y_pred = rf_model.predict(test_set)
print(classification_report(test_label, y_pred))