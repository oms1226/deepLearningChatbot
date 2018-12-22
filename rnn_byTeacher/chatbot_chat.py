# 챗봇 구현
# 골빈해커의 소스코드 수정본
# https://github.com/golbin/TensorFlow-Tutorials/tree/master/10%20-%20RNN/ChatBot

import tensorflow as tf
import numpy as np
import sys
import nltk
import chatbot_dialog       # 직접 구현한 파일
import chatbot_train        # 직접 구현한 파일


def predict(sess, logits, ph_enc_in, ph_dec_in, vocab, msg):
    tokens = nltk.regexp_tokenize(msg, '\w+')
    xx_enc = [vocab.index(t) if t in vocab else chatbot_dialog._UNK_ for t in tokens]
    xx_dec = []

    onehot_vector = np.eye(len(vocab))
    max_len_enc = len(xx_enc) * 2
    max_len_dec = 7

    for i in range(max_len_dec):
        # enc : (2, 164)
        # dec : (20, 164)
        enc, dec, _ = chatbot_dialog.transform(xx_enc, xx_dec, onehot_vector, max_len_enc, max_len_dec)

        # (1, 2, 164)와 (1, 20, 164)로 변환
        pred = sess.run(logits, {ph_enc_in: [enc], ph_dec_in: [dec]})

        pred_arg = np.argmax(pred, axis=2)      # (1, 20)
        pred_arg = np.reshape(pred_arg, -1)     # (20,)

        # 변환 결과에 종료 태그가 포함되면 종료
        if pred_arg[i] == chatbot_dialog._EOS_:
            break

        if pred_arg[i] not in chatbot_dialog._PRE_DEFINED_:
            xx_dec.append(pred_arg[i])

    idx = [vocab[i] for i in xx_dec]
    return ' '.join(idx).strip()


def run_chatbot(sess, logits, ph_enc_in, ph_dec_in, simulation=True):
    vocab = chatbot_dialog.load_vocab()

    if simulation:
        for line in ['안녕', '사막에서 뭐하니', '바람이 불고 있다']:
            pred = predict(sess, logits, ph_enc_in, ph_dec_in, vocab, line.strip())

            print('Q :', line)
            print('A :', pred)
            print()
    else:   # 터미널 입력
        while True:
            sys.stdout.write('Q : ')
            sys.stdout.flush()
            line = sys.stdin.readline()

            if 'exit' in line.lower():
                break

            pred = predict(sess, logits, ph_enc_in, ph_dec_in, vocab, line.strip())
            print('A :', pred)
            print()


def do_train_and_predict(simulation=True):
    logits, ph_enc_in, ph_dec_in, ph_targets = chatbot_train.build_model()
    sess = chatbot_train.do_train(logits, ph_enc_in, ph_dec_in, ph_targets, 501, False)

    run_chatbot(sess, logits, ph_enc_in, ph_dec_in, simulation)


def do_predict_by_checkpoint(simulation=True):
    logits, ph_enc_in, ph_dec_in, ph_targets = chatbot_train.build_model()

    # latest = 'model/conversation.ckpt-10000'      # 에러. 기존 모델과 호환 안됨
    latest = tf.train.latest_checkpoint('model')
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, latest)

    run_chatbot(sess, logits, ph_enc_in, ph_dec_in, simulation)


if __name__ == '__main__':
    # do_train_and_predict(simulation=True)
    # do_train_and_predict(simulation=False)

    do_predict_by_checkpoint(simulation=True)
    # do_predict_by_checkpoint(simulation=False)
