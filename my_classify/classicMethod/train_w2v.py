
# 初始化一个模型
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


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
            corpus_list.append(line.split('\t')[1].split())
    return label_list, corpus_list

data_dir = "../data/cnews/"
file_path = 'val_token.txt'
val_label, val_set = constructDataset(data_dir+file_path)
write_list = [data_dir+'train_token.txt', data_dir+'test_token.txt']

tarin_label, train_set = constructDataset(write_list[0]) # 50000
test_label, test_set = constructDataset(write_list[1]) # 10000
# 计算tf-idf
corpus_set = train_set + val_set + test_set # 全量计算tf-idf
# with open(data_dir+'corpus_set_token.txt', 'w', encoding='utf-8') as p:
#     p.writelines([' '.join(line)+'\n' for line in corpus_set])
#
# with open(data_dir+'vocab_set_token.txt', 'w', encoding='utf-8') as p:
#     lines = []
#     for line in corpus_set:
#         lines.extend(line)
#     p.writelines([i+'\n' for i in set(lines)])
#     print(len(set(lines)))
print(len(val_set))


# model = Word2Vec(corpus_set, size=100, window=5, min_count=1, workers=4)
model = Word2Vec(corpus_set, size=128, window=5, min_count=1, workers=4)
model.save(data_dir+"cnews_word2vec.model")