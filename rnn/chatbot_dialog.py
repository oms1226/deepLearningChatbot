# 데이터셋 생성
# train과 chat 파일에서 사용
import numpy as np
import nltk


PATH_SOURCE = 'Data/chat_source.txt'
PATH_VOCAB = 'Data/chat_vocab.txt'

# 패딩, 시작, 종료, 없음 태그
_PAD_, _STA_, _EOS_, _UNK_ = 0, 1, 2, 3
_PRE_DEFINED_ = ['_PAD_', '_STA_', '_EOS_', '_UNK_']


# 원본 파일을 읽어서 단어 파일 생성
def make_vocab():
    with open(PATH_SOURCE, 'r', encoding='utf-8') as fr:
        content = fr.read()
        #직접 토크나이즈를 생성하지 않고 nltk에서 가져다 써야 된다.
        words = nltk.regexp_tokenize(content, '\w+')
        words = sorted(set(words))

    with open(PATH_VOCAB, 'w') as fw:
        fw.write('\n'.join(words))


# 식별자를 숫자로 미리 정의해서 사용하고 있다. 문자열로 대체해 보자.
def load_vocab():
    f = open(PATH_VOCAB, 'r', encoding='utf-8')

    vocab = _PRE_DEFINED_ + [line.strip() for line in f]
    f.close()

    # 리스트에 포함된 index 함수 사용을 위해 numpy로 변환하지 않음
    return vocab


def get_vocab_size():
    return len(load_vocab())


def load_dialog():
    vocab = load_vocab()

    f = open(PATH_SOURCE, 'r', encoding='utf-8')

    dialogs = []
    for line in f:
        tokens = nltk.regexp_tokenize(line, '\w+')
        ids = [vocab.index(t) if t in vocab else _UNK_ for t in tokens]
        dialogs.append(ids)

    # 데이터 증강 : (질문과 대답) + (대답 + 질문)
    dialogs += [i.copy() for i in dialogs[1:]]
    dialogs += [i.copy() for i in dialogs[:1]]

    # 메모리가 공유되는지 확인
    assert len({id(p) for p in dialogs}) == len(dialogs)
    return dialogs, vocab

    #아래 코드는 기존 콜빈해커 코드인데 강사님이 위와 같이 수정하였다.
    # 메모리를 공유하기 때문에 문제 발생하는 코드
    # return dialogs + dialogs[1:] + dialogs[:1], vocab


def get_dataset():
    dialogs, vocab = load_dialog()
    assert len(dialogs) % 2 == 0

    # 디코더는 시작과 종료 태그가 있기 때문에 +1.
    max_len_enc = max([len(i) for i in dialogs[0::2]])
    max_len_dec = max([len(i) for i in dialogs[1::2]]) + 1

    onehot_vector = np.eye(len(vocab))

    xx_enc, xx_dec, target = [], [], []
    for i in range(0, len(dialogs), 2):
        enc, dec, tar = transform(dialogs[i], dialogs[i+1], onehot_vector, max_len_enc, max_len_dec)

        xx_enc.append(enc)
        xx_dec.append(dec)
        target.append(tar)

    return np.int32(xx_enc), np.int32(xx_dec), np.int32(target), vocab


def add_padding(seq, max_len, start=False, eos=False):
    # 인코더 입력(xx_enc)은 태그가 붙지 않는다.
    if start:
        seq = [_STA_] + seq         # xx_dec. 시작 태그
    elif eos:
        seq = seq + [_EOS_]         # target. 종료 태그

    if len(seq) < max_len:
        seq += [_PAD_] * (max_len - len(seq))

    return seq


# input : [38, 115]
# output : [125, 118, 37, 49]
def transform(input, output, onehot_vector, max_len_enc, max_len_dec):
    # 디코더의 입력과 출력 크기는 같다
    xx_enc = add_padding(input, max_len_enc)
    xx_dec = add_padding(output, max_len_dec, start=True)
    target = add_padding(output, max_len_dec, eos=True)

    # 구글 방식의 역순 입력
    # xx_enc.reverse()

    # xx_enc : [38, 115, 0, 0, 0, 0, 0, 0, 0]
    # xx_dec : [1, 125, 118, 37, 49, 0, 0, 0, 0, 0]
    # target : [125, 118, 37, 49, 2, 0, 0, 0, 0, 0]
    return onehot_vector[xx_enc], onehot_vector[xx_dec], target


if __name__ == '__main__':
    # make_vocab()
    # dialogs, vocab = load_dialog()
    # print(*dialogs, sep='\n')

    x1, x2, y, vocab = get_dataset()
    print(x1.shape, x2.shape, y.shape)
    print(*y, sep='\n')
