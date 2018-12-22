import tensorflow as tf
import numpy as np
import chatbot_dialog


def build_model():
    vocab_size = chatbot_dialog.get_vocab_size()
    n_hidden = 128
    n_layers = 3

    ph_enc_in = tf.placeholder(tf.float32, [None, None, vocab_size])
    ph_dec_in = tf.placeholder(tf.float32, [None, None, vocab_size])
    ph_targets = tf.placeholder(tf.int64, [None, None])

    # 인코더. scope가 없으면 변수 이름 충돌. 그 자체로 영역을 구분하기 때문에 유지.
    with tf.variable_scope('encode'):
        cells = [tf.nn.rnn_cell.BasicLSTMCell(n_hidden) for _ in range(n_layers)]
        enc_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, ph_enc_in, dtype=tf.float32)

    # 디코더. 레이어의 갯수는 3개.
    # 초기값으로 인코더의 내부 상태를 가리키는 enc_states를 넘겨주고 있다.
    with tf.variable_scope('decode'):
        cells = [tf.nn.rnn_cell.BasicLSTMCell(n_hidden) for _ in range(n_layers)]
        dec_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, ph_dec_in, dtype=tf.float32,
                                                initial_state=enc_states)

    time_steps = tf.shape(outputs)[1]
    outputs = tf.reshape(outputs, [-1, n_hidden])

    logits = tf.layers.dense(outputs, units=vocab_size, activation=None)
    logits = tf.reshape(logits, [-1, time_steps, vocab_size])

    return logits, ph_enc_in, ph_dec_in, ph_targets


def do_train(logits, ph_enc_in, ph_dec_in, ph_targets, loop_count=501, will_save=False):
    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ph_targets)
    loss = tf.reduce_mean(loss_i)
 m  
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    xx_enc, xx_dec, target, vocab = chatbot_dialog.get_dataset()

    for i in range(loop_count):
        sess.run(train, {ph_enc_in: xx_enc, ph_dec_in: xx_dec, ph_targets: target})

        if i % 10 == 0:
            print(i, sess.run(loss, {ph_enc_in: xx_enc, ph_dec_in: xx_dec, ph_targets: target}))
    print('-' * 50)

    if will_save:
        saver = tf.train.Saver()
        saver.save(sess, 'model/chatbot', global_step=501)

    return sess


def do_predict(sess, logits, ph_enc_in, ph_dec_in, xx_enc, xx_dec, target, vocab):
    pred = sess.run(logits, {ph_enc_in: xx_enc, ph_dec_in: xx_dec})
    pred_arg = np.argmax(pred, axis=2)

    # 패딩을 제외한 정확도 계산
    total, answer = 0, 0
    for p, t in zip(pred_arg, target):
        equals = np.where(t == chatbot_dialog._EOS_)      # 인덱스 찾기
        pos = np.reshape(equals, -1)[0] + 1     # 배열이기 때문에 0번째 추출. 찾은 위치 다음이니까 +1.

        total += pos                            # 글자 전체 갯수
        answer += np.sum(t[:pos] == p[:pos])    # 맞은 것만 누적
        print(np.vstack([t[:pos], p[:pos]]))    # 맞은 위치 비교
        print()
    print('-' * 50)

    # 기존 코드는 패딩까지도 정확도에 포함하고 있다.
    # 패딩을 빼고 계산하면 정확도가 조금 떨어질 수밖에 없다.
    print('acc1 :', np.mean(pred_arg == target))
    print('acc2 :', answer / total, answer, total)
    # acc1 : 0.9961538461538462
    # acc2 : 0.992619926199262 269 271

    # 출력 결과 보여줄 것. 정확도가 뭔지 느낌이 없을 수 있다.
    # print(pred_arg == target)
    # [[False  True  True  True  True  True  True  True  True  True]
    #  [ True  True  True  True  True  True  True  True  True  True]
    #  [ True  True  True  True  True  True  True  True  True  True]

    def convert(value, vocab_np):
        equals = np.where(value == 2)
        pos = np.reshape(equals, -1)[0]
        idx = value[:pos]
        return vocab_np[idx]

    vocab_np = np.array(vocab)

    print('정답 :', convert(target[1], vocab_np))
    print('예측 :', convert(pred_arg[1], vocab_np))
    print()

    print('정답 :', convert(target[2], vocab_np))
    print('예측 :', convert(pred_arg[2], vocab_np))

    # 정답 : ['넌' '누구지' '참' '예쁘구나']
    # 예측 : '넌' '누구지' '참' '예쁘구나']
    #
    # 정답 : ['이리' '와서' '나하고' '놀자']
    # 예측 : ['이리' '와서' '나하고' '놀자']


# 학습과 예측
def do_train_and_predict():
    logits, ph_enc_in, ph_dec_in, ph_targets = build_model()
    sess = do_train(logits, ph_enc_in, ph_dec_in, ph_targets)

    xx_enc, xx_dec, target, vocab = chatbot_dialog.get_dataset()
    do_predict(sess, logits, ph_enc_in, ph_dec_in, xx_enc, xx_dec, target, vocab)


# 학습없이 예측만. 체크포인트 파일 사용
def do_predict_by_checkpoint():
    logits, ph_enc_in, ph_dec_in, ph_targets = build_model()

    latest = tf.train.latest_checkpoint('model')
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, latest)

    xx_enc, xx_dec, target, vocab = chatbot_dialog.get_dataset()
    do_predict(sess, logits, ph_enc_in, ph_dec_in, xx_enc, xx_dec, target, vocab)


if __name__ == '__main__':
    # do_train_and_predict()
    do_predict_by_checkpoint()
