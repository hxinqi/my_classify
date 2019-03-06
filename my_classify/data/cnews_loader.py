#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os
import codecs
from gensim.models import Word2Vec
from sklearn.externals import joblib

def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return codecs.open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(list(zip(words, list(range(len(words))))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经','房产','家居',
        '教育', '科技', '时尚', '时政', '游戏','娱乐']
    cat_to_id = dict(list(zip(categories, list(range(len(categories))))))
    categories = ['sports', 'finance', 'house', 'living',
        'education', 'tech', 'fashion', 'policy', 'game', 'entertaiment']

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def my_read_file(path):
    """
    path: file path
    rtype: lable_list and corpus_list
    """
    label_list = []
    corpus_list = []
    with open(path, 'r', encoding='utf-8') as p:
        for line in p.readlines():
            label_list.append(line.split('\t')[0])
            corpus_list.append(line.split('\t')[1].split())
    return label_list, corpus_list

def my_process_file(filename, cat_to_id, max_length=600):
    """将文件转换为w2v表示"""
    labels, contents = my_read_file(filename)

    data_dir = "E:\\Practice\\myworkspace\\paper\\\my_classify\\data\\cnews\\"
    model = Word2Vec.load(data_dir + "cnews_word2vec.model")

    w2v, label_id = [], []
    for i in range(len(contents)):
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        w2v.append([model[x] for x in contents[i]])
        label_id.append(cat_to_id[labels[i]])
    # def get_train_set(w2v):
    #     x_train = []
    #     for line in w2v:
    #         tmp = []
    #         for i in line:
    #             tmp.extend(i)
    #         x_train.append(tmp)
    #     return x_train
    #
    # w2v = get_train_set(w2v)
    def my_pad_set(sequences,
                      maxlen=None,
                      dtype='float32',):
        lengths = []
        for x in sequences:
            lengths.append(len(x))

        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)
        x = np.ones((num_samples, maxlen,5)).astype(dtype)
        for idx, s in enumerate(sequences):
            if not len(s):  # pylint: disable=g-explicit-length-test
                continue  # empty list/array was found
            trunc = s[:maxlen]
            trunc = np.asarray(trunc, dtype=dtype)
            x[idx, :len(trunc)] = trunc
        return x
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # x_pad = kr.preprocessing.sequence.pad_sequences(w2v, max_length)
    x_pad = my_pad_set(w2v, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def my1_process_file(filename, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    labels, contents = my_read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        # w2v.append([model[x] for x in contents[i]])
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = contents
    # x_pad = kr.preprocessing.sequence.pad_sequences(contents, max_length,dtype=str)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def my_batch_iter(x, y, batch_size=64, max_length=600):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    x_shuffle = []
    y_shuffle = []
    for index in indices:
        x_shuffle.append(x[index])
        y_shuffle.append(y[index])

    data_dir = "E:\\Practice\\myworkspace\\paper\\\my_classify\\data\\cnews\\"
    model = Word2Vec.load(data_dir + "cnews_word2vec.model")

    # voca_dict_tf=joblib.load(tmp_catalog + "voca_dict_tf.pkl")
    idf_dict = joblib.load(data_dir + "idf_dict.pkl")
    voca_matrix_tf = joblib.load(data_dir + "voca_matrix_tf.pkl")
    tf_idf_matrix = joblib.load(data_dir + "tf_idf_dict.pkl")
    voc2id_dict = joblib.load(data_dir + "voc2id_dict.pkl")
    j = 0
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        w2v = []
        for contents in x_shuffle[start_id:end_id]:
            # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

            line_join_vec = []
            tf_m = voca_matrix_tf[indices[j], :]
            tf_vec=tf_m.toarray()[0]
            tf_idf_m = tf_idf_matrix[indices[j], :]
            tf_idf_vec=tf_idf_m.toarray()[0]
            j += 1
            for x in contents:
                tmp_join_vec = []
                tmp_join_vec.extend(model[x])
                tmp_join_vec.append(tf_vec[voc2id_dict[x]])
                tmp_join_vec.append(idf_dict[x])
                tmp_join_vec.append(tf_idf_vec[voc2id_dict[x]])
                line_join_vec.append(tmp_join_vec)

            w2v.append(line_join_vec)

            # w2v.append([model[x] for x in contents])

        def my_pad_set(sequences,
                       maxlen=None,
                       dtype='float32', ):
            lengths = []
            for x in sequences:
                lengths.append(len(x))

            num_samples = len(sequences)
            if maxlen is None:
                maxlen = np.max(lengths)
            x = np.ones((num_samples, maxlen, 131)).astype(dtype)
            for idx, s in enumerate(sequences):
                if not len(s):  # pylint: disable=g-explicit-length-test
                    continue  # empty list/array was found
                trunc = s[:maxlen]
                trunc = np.asarray(trunc, dtype=dtype)
                x[idx, :len(trunc)] = trunc
            return x

        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = my_pad_set(w2v, max_length)
        # x_pad = kr.preprocessing.sequence.pad_sequences(w2v, max_length, dtype=str)
        # yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        yield x_pad, y_shuffle[start_id:end_id]

def my_batch_iter_no(x, y, batch_size=64, max_length=600):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    # x_shuffle = x[indices]
    # y_shuffle = y[indices]
    # x_shuffle = []
    x_shuffle = x
    y_shuffle = y
    # y_shuffle = []
    # for index in indices:
    #     x_shuffle.append(x[index])
    #     y_shuffle.append(y[index])

    data_dir = "E:\\Practice\\myworkspace\\paper\\\my_classify\\data\\cnews\\"
    model = Word2Vec.load(data_dir + "cnews_word2vec.model")

    # voca_dict_tf=joblib.load(tmp_catalog + "voca_dict_tf.pkl")
    idf_dict = joblib.load(data_dir + "idf_dict.pkl")
    voca_matrix_tf = joblib.load(data_dir + "voca_matrix_tf.pkl")
    tf_idf_matrix = joblib.load(data_dir + "tf_idf_dict.pkl")
    voc2id_dict = joblib.load(data_dir + "voc2id_dict.pkl")
    j = 0
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        w2v = []
        for contents in x_shuffle[start_id:end_id]:
            # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

            line_join_vec = []
            tf_m = voca_matrix_tf[indices[j], :]
            tf_vec=tf_m.toarray()[0]
            tf_idf_m = tf_idf_matrix[indices[j], :]
            tf_idf_vec=tf_idf_m.toarray()[0]
            j += 1
            for x in contents:
                tmp_join_vec = []
                tmp_join_vec.extend(model[x])
                tmp_join_vec.append(tf_vec[voc2id_dict[x]])
                tmp_join_vec.append(idf_dict[x])
                tmp_join_vec.append(tf_idf_vec[voc2id_dict[x]])
                line_join_vec.append(tmp_join_vec)

            w2v.append(line_join_vec)

            # w2v.append([model[x] for x in contents])

        def my_pad_set(sequences,
                       maxlen=None,
                       dtype='float32', ):
            lengths = []
            for x in sequences:
                lengths.append(len(x))

            num_samples = len(sequences)
            if maxlen is None:
                maxlen = np.max(lengths)
            x = np.ones((num_samples, maxlen, 131 )).astype(dtype)
            for idx, s in enumerate(sequences):
                if not len(s):  # pylint: disable=g-explicit-length-test
                    continue  # empty list/array was found
                trunc = s[:maxlen]
                trunc = np.asarray(trunc, dtype=dtype)
                x[idx, :len(trunc)] = trunc
            return x

        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = my_pad_set(w2v, max_length)
        # x_pad = kr.preprocessing.sequence.pad_sequences(w2v, max_length, dtype=str)
        # yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        yield x_pad, y_shuffle[start_id:end_id]



