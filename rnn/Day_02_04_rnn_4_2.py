#Day_02_04_rnn_4_2.py
import numpy as np
import tensorflow as tf


def make_onshot_basic(text):
    idx2char = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(idx2char)}

    print(idx2char)#['e', 'n', 'o', 'r', 's', 't']
    print(char2idx)#{'e': 0, 'n': 1, 'o': 2, 'r': 3, 's': 4, 't': 5}

    text_idx = [c for c in text]
    print(text_idx)
    text_idx = [char2idx[c] for c in text]
    print(text_idx)

    x = text_idx[:-1]
    y = text_idx[1:]
    print(x)
    print(y)

    eye = np.eye(len(char2idx), dtype=np.int32)#단위행렬
    print(eye)
    print()

    xx = eye[x]#x 값을 2차원 행렬로 변경함
    print(xx)

    return np.int32([xx]), tf.constant([y]), np.array(idx2char)


make_onshot_basic('tensor')






print('\n\n\n\n\n\n\n\n')