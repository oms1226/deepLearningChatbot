# Day_03_05_seq2seq.py
import tensorflow as tf
import numpy as np

def make_vocab_and_dict(data):
    eng = sorted({c for w, _ in data for c in w})
    kor = sorted({c for _, w in data for c in w})

    print(eng)
    print(kor)
    # ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 'u', 'w']
    # ['나', '등', '람', '랑', '무', '바', '식', '영', '웅', '음', '전', '파']

    # S(start), E(end), P(padding)
    vocab = ''.join(eng + kor + list('SEP'))
    print(vocab)
    #abdefhilmnopruw나등람랑무바식영웅음전파SEP

    char2idx = {c: i for i, c in enumerate(vocab)}
    print(char2idx)
    #{'a': 0, 'b': 1, 'd': 2, 'e': 3, 'f': 4, 'h': 5, 'i': 6, 'l': 7, 'm': 8, 'n': 9, 'o': 10, 'p': 11, 'r': 12, 'u': 13, 'w': 14, '나': 15, '등': 16, '람': 17, '랑': 18, '무': 19, '바': 20, '식': 21, '영': 22, '웅': 23, '음': 24, '전': 25, '파': 26, 'S': 27, 'E': 28, 'P': 29}
    ##vocab 을 통해 index 를 찾을 수 있지만, 성능상에 이유로 dict으로 같이 구성하는게 좋다.
    return vocab, char2idx

def make_batch(data, char2idx):
    eye = np.eye(len(char2idx))

    inputs, outputs, targets = [], [], []
    for eng, kor in data:
        #print(eng, 'S'+kor, kor+'E')
        # food S음식 음식E
        # wood S나무 나무E
        # blue S파랑 파랑E
        # lamp S전등 전등E
        # wind S바람 바람E
        # hero S영웅 영웅E
        input = [char2idx[i] for i in eng]
        output = [char2idx[i] for i in 'S'+kor]
        target = [char2idx[i] for i in kor+'E']
        print(input, output, target)
        #[4, 10, 10, 2] [27, 24, 21] [24, 21, 28]
        # [14, 10, 10, 2] [27, 15, 19] [15, 19, 28]
        # [1, 7, 13, 3] [27, 26, 18] [26, 18, 28]
        # [7, 0, 8, 11] [27, 25, 16] [25, 16, 28]
        # [14, 6, 9, 2] [27, 20, 17] [20, 17, 28]
        # [5, 3, 12, 10] [27, 22, 23] [22, 23, 28]
        inputs.append(eye[input])
        outputs.append(eye[output])
        targets.append(target)
    return np.float32(inputs), np.float32(outputs), np.float32(targets)

def show_seq2seq(data, char2idx, vocab, pred_list):
    input_data, output_data, target_data = make_batch(data, char2idx)

    n_classes, n_hidden = len(vocab), 128

    ph_enc_in = tf.placeholder(tf.float32, [None, None, n_classes])
    ph_dec_in = tf.placeholder(tf.float32, [None, None, n_classes])
    ph_target = tf.placeholder(tf.int32, [None, None])
    #tf.nn.rnn_cell.BasicRNNCell 를 두개 이상 선언할때는 name 으로 구분하여 tensorflow 내부적으로 충돌이 없다.
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, name='enc_cell')
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, ph_enc_in, dtype=tf.float32)

    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, name='dec_cell')
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, ph_dec_in, dtype=tf.float32,
                                           initial_state=enc_states)

    z = tf.layers.dense(outputs, n_classes)
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=ph_target)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_enc_in: input_data, ph_dec_in: output_data, ph_target: target_data})
        print(i, sess.run(loss, {ph_enc_in: input_data, ph_dec_in: output_data, ph_target: target_data}))
    print('-' * 50)

    print("data:", data)
    print("data[0][1]:", data[0][1])
    #항상 동일한 글자 갯수가 동일하기에 "data[0][1]: 음식" 로 정해도 무방하다.
    new_data = [(w, 'P' * len(data[0][1])) for w in pred_list]
    print(new_data)

    input_data, output_data, target_data = make_batch(new_data, char2idx)
    pred = sess.run(z, {ph_enc_in: input_data, ph_dec_in: output_data, ph_target: target_data})

    pred_arg = np.argmax(pred, axis=2)
    print(pred_arg)

    results = [[vocab[j] for j in i] for i in pred_arg]
    print(results)

    print([''.join(w[:-1]) for w in results])

    sess.close()

data = [
    ['food', '음식'],
    ['wood', '나무'],
    ['blue', '파랑'],
    ['lamp', '전등'],
    ['wind', '바람'],
    ['hero', '영웅']
]
#해당 예제는 영어 4글자, 한글 2글자로 된 seq2seq 예제이다. 그래서 다양한 글자길이를 쓰려면, 아래를 참조해서 추가 작업을 해야 된다!!!
##/Users/oms1226/_workspace/deepLearningChatbot/rnn/Day_02_05_rnn_5_different.py
vocab, char2idx = make_vocab_and_dict(data)
show_seq2seq(data, char2idx, vocab, ['blue', 'hero'])