#Day_03_04_word2vec_adv.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_vocab_and_dict(corpus, stop_words):#stop_words  불용어라는 뜻이다.
    def remove_stop_words(corpus, stop_words):
        return [[word for word in sent.split() if word not in stop_words] for sent in corpus]

    corpus_by_word = remove_stop_words(corpus, stop_words)
    # print(*corpus_by_word, sep='\n')
    # ['king', 'strong', 'man']
    # ['queen', 'wise', 'woman']
    # ['boy', 'young', 'man']
    # ['girl', 'young', 'woman']
    # ['prince', 'young', 'king']
    # ['princess', 'young', 'queen']
    # ['man', 'strong']
    # ['woman', 'pretty']
    # ['prince', 'boy', 'king']
    # ['princess', 'girl', 'queen']

    def make_vocabulary(corpus_by_word):
        # return [word for sent in corpus_by_word for word in sent]
        #return {word for sent in corpus_by_word for word in sent}#중복제거
        return sorted({word for sent in corpus_by_word for word in sent})#정렬

    vocab = make_vocabulary(corpus_by_word)#1차원 단어들로 추출
    # print(vocab)
    # {'man', 'strong', 'woman', 'young', 'pretty', 'wise', 'girl', 'prince', 'boy', 'queen', 'princess', 'king'}

    corpus_idx = [[vocab.index(word) for word in sent] for sent in corpus_by_word]
    # print(corpus_idx)#각 말뭉치의 그룹을 깨지 않으면서 index 로 변경했다.
    # [[2, 8, 3], [7, 9, 10], [0, 11, 3], [1, 11, 10], [5, 11, 2], [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]

    return vocab, corpus_idx

corpus = [
    'king is a strong man',
    'queen is a wise woman',
    'boy is a young man',
    'girl is a young woman',
    'prince is a young king',
    'princess is a young queen',
    'man is strong',
    'woman is pretty',
    'prince is a boy will be king',
    'princess is a girl will be queen'
]

make_vocab_and_dict(corpus, ['is', 'a', 'will', 'be'])