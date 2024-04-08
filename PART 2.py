import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math
import chardet
import codecs

import os

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return stopwords

def remove_stopwords(text, stopwords):
    words = text.split()
    filtered_text = [word for word in words if word not in stopwords]
    return ' '.join(filtered_text)



# 读取并预处理中文语料库
def preprocess_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
        # 去除标点符号（可以根据需要调整正则表达式）
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        stopwords_file_path = 'cn_stopwords.txt'
        cn_stopwords = load_stopwords(stopwords_file_path)
        
        text = remove_stopwords(text, cn_stopwords)
        # 使用jieba进行分词
        words = jieba.lcut(text)    
        return words,text 

# 计算词频并排序
def calculate_word_frequencies(words):
    counter = Counter(words)
    
    # 按频次降序排列
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words

def calculate_average_entropy(text, unit='word'):
    if unit == 'word':
        words = list(jieba.cut(text))  # 分词
    elif unit == 'char':
        words = list(text)  # 按字切分
    else:
        raise ValueError("Invalid unit. Choose either 'word' or 'char'.")

    word_counts = Counter(words)
    total_words = sum(word_counts.values())

    total_entropy = sum(-count / total_words * math.log2(count / total_words) for count in word_counts.values())
    average_entropy = total_entropy / len(word_counts)

    return average_entropy
# 绘制词频-排名图
def plot_zipf_law(word_freqs, max_rank=1000):
    ranks = range(1, max_rank + 1)
    frequencies = [freq for _, freq in word_freqs[:max_rank]]

    theoretical_freqs = []
    for rank in ranks:
        if rank == 1:
            theoretical_freqs.append(1)  # 直接设置rank=1时的理论词频为1
        else:
            theoretical_freqs.append(1 / math.log(rank, 10))

    plt.loglog(ranks, frequencies, 'o', markersize=3, alpha=0.5)
    plt.loglog(ranks, theoretical_freqs, label='Theoretical Zipf Law (α=1)', color='r', linestyle='--')

    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf Law for Chinese Corpus')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    book_titles_list = "白马啸西风,碧血剑,飞狐外传,连城诀,鹿鼎记,三十三剑客图,射雕英雄传,神雕侠侣,书剑恩仇录,天龙八部,侠客行,笑傲江湖,雪山飞狐,倚天屠龙记,鸳鸯刀,越女剑"#
    output_file_path = 'jyxstxtqj_downcc.com\\all.txt'
    for book_title in book_titles_list.split(','):
        book_title = book_title.strip()  # 去除可能存在的多余空白字符
        file_path='jyxstxtqj_downcc.com\{}.txt'.format(book_title)
        merged_content = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            merged_content += f.read()
        # 保存合并后的内容到新的文本文件
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        corpus_file = file_path
        words,text = preprocess_corpus(corpus_file)
        #word_freqs = calculate_word_frequencies(words)
        
        #plot_zipf_law(word_freqs)
        # 计算以词为单位的平均信息熵
        average_word_entropy = calculate_average_entropy(text, unit='word')
        print(f"Average entropy (word level): {average_word_entropy:.4f} bits")

        # 计算以字为单位的平均信息熵
        average_char_entropy = calculate_average_entropy(text, unit='char')
        print(f"Average entropy (character level): {average_char_entropy:.4f} bits")