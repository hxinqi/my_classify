"""
val数据集最短...所有调通它的分词后
我们接下来就要考虑，将train和test文件也分词处理
保存各自的token文件到本地

多进程处理
"""
import multiprocessing
from tools.tokenizer.wordCut import WordCut

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

if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=4)
    for file_path, write_path in zip(file_list, write_list):
        pool.apply_async(tokenFile, (file_path, write_path, ))
    pool.close()
    pool.join() # 调用join()之前必须先调用close()
    print("Sub-process(es) done.")
