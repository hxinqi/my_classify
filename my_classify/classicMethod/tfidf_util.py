from collections import Mapping, defaultdict
from sklearn.externals import joblib
import numpy as np
import array
import scipy.sparse as sp
from sklearn.externals import six
from operator import  itemgetter
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



def build_vocab():
    '''建立词库'''
    tmp_catalog = '../data/cnews/'
    write_list = [tmp_catalog + 'train_token.txt', tmp_catalog + 'test_token.txt']

    tarin_label, train_set = constructDataset(write_list[0])  # 50000
    test_label, test_set = constructDataset(write_list[1])  # 10000
    file_path = 'val_token.txt'
    val_label, val_set = constructDataset(tmp_catalog + file_path)
    # 计算tf-idf
    corpus_set = train_set + val_set + test_set  # 全量计算tf-idf
    line = list()
    i = 0
    for text in corpus_set:
        text_set=set(text.split())
        line.extend(list(text_set))
        i=i+1
        if i == 1000:
            i=0
            tmp=list(set(line))
            line = tmp
    voca = list(set(line))
    voca_dict = {voca[i]:i for i in range(len(voca))}

    tmp_catalog = '../data/cnews/'
    joblib.dump(voca,tmp_catalog + "word_list.pkl")
    return voca_dict

def count_vec(raw_documents):
    '''词数量统计'''
    tmp_catalog = '../data/cnews/'
    voca_list=joblib.load(tmp_catalog + "word_list.pkl")
    voca_dict = {voca_list[i]: 0 for i in range(len(voca_list))}
    for doc in raw_documents:
        for word in doc.split():
            voca_dict[word]+=1
    tmp_catalog = '../data/cnews/'
    joblib.dump(voca_dict,tmp_catalog + "voca_dict.pkl")
    return voca_dict

def get_tf_vec():
    '''得到词频'''
    tmp_catalog = '../data/cnews/'
    voca_dict=joblib.load(tmp_catalog + "voca_dict.pkl")
    word_num=len(voca_dict)
    for k, v in voca_dict.items():
        voca_dict[k]/= word_num
    joblib.dump(voca_dict,tmp_catalog + "voca_dict_tf.pkl")
    return voca_dict

def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))

def count_vocab(raw_documents):
    """Create sparse feature matrix, and vocabulary where fixed_vocab=False
    """
    vocabulary = defaultdict()
    vocabulary.default_factory = vocabulary.__len__

    j_indices = []
    indptr = _make_int_array()
    values = _make_int_array()
    indptr.append(0)
    for doc in raw_documents:
        feature_counter = {}
        for feature in doc.split():
            try:
                feature_idx = vocabulary[feature]
                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1
            except KeyError:
                # Ignore out-of-vocabulary items for fixed_vocab=True
                continue

        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))
    j_indices = np.asarray(j_indices, dtype=np.intc)
    indptr = np.frombuffer(indptr, dtype=np.intc)
    values = np.frombuffer(values, dtype=np.intc)
    vocabulary = dict(vocabulary)
    X = sp.csr_matrix((values, j_indices, indptr),
                      shape=(len(indptr) - 1, len(vocabulary)),
                      dtype=np.float64)
    tmp_catalog = '../data/cnews/'
    joblib.dump(vocabulary,tmp_catalog + "voc2id_dict.pkl")
    joblib.dump(X , tmp_catalog + "csr_matrix_count.pkl")
    return vocabulary, X
def get_tf_matrix(raw_documents):
    '''得到词频矩阵'''
    tmp_catalog = '../data/cnews/'
    csr_matrix_count=joblib.load(tmp_catalog + "csr_matrix_count.pkl")
    i=0
    for doc in raw_documents:
        csr_matrix_count[i]=csr_matrix_count[i]/len(doc.split())
        i+=1
    joblib.dump(csr_matrix_count,tmp_catalog + "voca_matrix_tf.pkl")
    return csr_matrix_count
def get_idf_vec(raw_documents):
    '''得到逆文档频率
      idf(d, t) = log [ n / (df(d, t) + 1) ])
      '''
    tmp_catalog = '../data/cnews/'
    counter_matrix=joblib.load(tmp_catalog + "csr_matrix_count.pkl")
    df = np.bincount(counter_matrix.indices, minlength=counter_matrix.shape[1])
    n_samples, n_features = counter_matrix.shape
    idf = np.log(float(n_samples) / df+1.0)
    voc2id_dict=joblib.load( tmp_catalog + "voc2id_dict.pkl")
    for w in voc2id_dict:
        voc2id_dict[w]=idf[voc2id_dict[w]]
    joblib.dump(voc2id_dict, tmp_catalog + "idf_dict.pkl")
    joblib.dump(idf,tmp_catalog + "idf_array.pkl")
    return voc2id_dict

def get_tf_idf_vec():
    '''得到词库的词频逆文档频率矩阵'''
    tmp_catalog = '../data/cnews/'
    voca_matrix_tf = joblib.load(tmp_catalog + "voca_matrix_tf.pkl")
    idf_dict = joblib.load(tmp_catalog + "idf_dict.pkl")
    voc2id_dict=joblib.load(tmp_catalog + "voc2id_dict.pkl")

    for t, i in sorted(six.iteritems(voc2id_dict),key=itemgetter(1)):
        voca_matrix_tf[:, i]=voca_matrix_tf[:,i].multiply(idf_dict[t])
    print('结束。。。。')
    joblib.dump(voca_matrix_tf, tmp_catalog + "tf_idf_dict.pkl")
    return voca_matrix_tf

def my_test():
    tmp_catalog = '../data/cnews/'
    write_list = [tmp_catalog + 'train_token.txt', tmp_catalog + 'test_token.txt']
    tarin_label, train_set = constructDataset(write_list[0])  # 50000
    test_label, test_set = constructDataset(write_list[1])  # 10000
    file_path = 'val_token.txt'
    val_label, val_set = constructDataset(tmp_catalog + file_path)
    # 计算tf-idf
    corpus_set = train_set + val_set + test_set  # 全量计算tf-idf
    # count_vec(corpus_set)
    # get_tf_vec()
    # idf_dict=get_idf_vec(corpus_set)
    # count_vocab(corpus_set)
    # get_idf_vec(corpus_set)
    # get_tf_matrix(corpus_set)
if __name__ == '__main__':
    # voca=build_vocab()
    # print(len(voca))
    # my_test()
    get_tf_idf_vec()
