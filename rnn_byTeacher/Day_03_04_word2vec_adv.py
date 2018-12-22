# Day_03_04_word2vec_adv.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_vocab_and_dict(corpus, stop_words):
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
        return sorted({word for sent in corpus_by_word for word in sent})

    vocab = make_vocabulary(corpus_by_word)
    # print(vocab)
    # ['boy', 'girl', 'king', 'man', 'pretty', 'prince', 'princess', 'queen', 'strong', 'wise', 'woman', 'young']

    corpus_idx = [[vocab.index(word) for word in sent] for sent in corpus_by_word]
    print(corpus_idx)
    # [[2, 8, 3], [7, 9, 10], [0, 11, 3], [1, 11, 10], [5, 11, 2], [6, 11, 7], [3, 8], [10, 4], [5, 0, 2], [6, 1, 7]]

    return vocab, corpus_idx


def extract_excluded(tokens, target, window_size, token_count):
    start = max(target - window_size, 0)
    end = min(target + window_size, token_count - 1) + 1

    # [3, 4, 5, 6, 7] : 5 -> [3, 4, 6, 7]
    return [tokens[i] for i in range(start, end) if i != target]


def build_dataset(corpus_idx, n_classes, window_size, is_skipgram):
    xx, yy = [], []
    for sent in corpus_idx:
        for i, target in enumerate(sent):
            # i : 중심 단어
            # ctx : 주변 단어들
            ctx = extract_excluded(sent, i, window_size, len(sent))
            # print(ctx)

            if is_skipgram:
                for neighbor in ctx:
                    xx.append(target)
                    yy.append(neighbor)
                    # print(target, neighbor)
            else:
                xx.append(ctx)
                yy.append(target)

    return make_onehot(xx, yy, n_classes, is_skipgram)


def make_onehot(xx, yy, n_classes, is_skipgram):
    x = np.zeros([len(xx), n_classes], dtype=np.float32)
    y = np.zeros([len(yy), n_classes], dtype=np.float32)

    for i, (input, label) in enumerate(zip(xx, yy)):
        y[i, label] = 1

        if is_skipgram:
            x[i, input] = 1
        else:
            # [[0 0 1 0]
            #  [1 0 0 0]
            #  [0 0 0 1]]
            # -----------
            # [0.33 0.0 0.33 0.33]
            z = np.zeros([len(input), n_classes], dtype=np.float32)
            for j, pos in enumerate(input):
                z[j, pos] = 1
            x[i] = np.mean(z, axis=0)

    return x, y


def show_word2vec(corpus, window_size, is_skipgram):
    vocab, corpus_idx = make_vocab_and_dict(corpus, ['is', 'a', 'will', 'be'])
    n_classes = len(vocab)
    n_embeds = 2

    # (52, 12) (52, 12)
    x, y = build_dataset(corpus_idx, n_classes, window_size, is_skipgram)
    print(x.shape, y.shape)

    w_hidden = tf.Variable(tf.random_uniform([n_classes, n_embeds]))
    hidden_layer = tf.matmul(x, w_hidden)

    z = tf.layers.dense(hidden_layer, n_classes)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        sess.run(train)
        if i % 1000 == 0:
            print(i, sess.run(loss))
    print('-' * 50)

    vectors = sess.run(w_hidden)
    print(vectors)
    sess.close()

    for word, (x1, x2) in zip(vocab, vectors):
        plt.text(x1, x2, word)

    ax_min = np.min(vectors, axis=0) - 1
    ax_max = np.max(vectors, axis=0) + 1

    plt.xlim(ax_min[0], ax_max[0])
    plt.ylim(ax_min[1], ax_max[1])

    plt.show()


corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']

# show_word2vec(corpus, 2, True)
show_word2vec(corpus, 2, False)
