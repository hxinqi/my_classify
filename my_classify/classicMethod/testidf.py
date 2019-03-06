from sklearn.externals import joblib
tmp_catalog = '../data/cnews/'
# tfidf=joblib.load(tmp_catalog+"tfidf.pkl")
# vectorizer=joblib.load(tmp_catalog+"vectorizer.pkl")
# print(vectorizer.vocabulary_)
# joblib.dump(tfidf,tmp_catalog+"tfidf.pkl")
# joblib.dump(vectorizer,tmp_catalog+"vectorizer.pkl")

# from gensim.models import Word2Vec
#
# data_dir = "../data/cnews/"
# model = Word2Vec.load(data_dir+"cnews_word2vec.model")
# # print(model['体育'])
# print (model.wv.doesnt_match(u"体育 篮球 出访".split()))
# print(model.wv.vocab)

# tfidf向量保存
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

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

tmp_catalog = '../data/cnews/'
write_list = [tmp_catalog+'train_token.txt', tmp_catalog+'test_token.txt']

tarin_label, train_set = constructDataset(write_list[0]) # 50000
test_label, test_set = constructDataset(write_list[1]) # 10000
file_path = 'val_token.txt'
val_label, val_set = constructDataset(tmp_catalog+file_path)
# 计算tf-idf
corpus_set = train_set + val_set + test_set # 全量计算tf-idf
print("length of corpus is: " + str(len(corpus_set)))
vectorizer = CountVectorizer() # drop df < 1e-5,去低频词
# vectorizer = CountVectorizer(min_df=1e-5) # drop df < 1e-5,去低频词
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_set))
joblib.dump(tfidf,tmp_catalog+"tfidf.pkl")
joblib.dump(vectorizer,tmp_catalog+"vectorizer.pkl")
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

# # 随机森林分类器
# from sklearn.ensemble import RandomForestClassifier
#
#
# rf_model = RandomForestClassifier(n_estimators=200, random_state=1080)
# rf_model.fit(train_set, train_label)
# print("val mean accuracy: {0}".format(rf_model.score(val_set, val_label)))
# y_pred = rf_model.predict(test_set)
# print(classification_report(test_label, y_pred))