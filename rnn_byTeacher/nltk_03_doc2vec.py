# 아래 사이트에서 [1.3 Document Classification] 정리한 문서
# https://www.nltk.org/book/ch06.html

# 파일 단위의 문서를 긍정 또는 부정으로 분류하기 위해 문서 피쳐를 생성하는 방법에 대해 설명한다.
# [6. Learning to Classify Text]에 포함된 [1. Supervised Classification] 내용의 일부.

# nltk_basic.txt 수업 후에 진행 (예상 시간 : 2시간)

# 순서 및 흐름
# 1. 리뷰에 포함된 모든 단어의 빈도 분포를 계산해서 빈도가 높은 단어 피쳐 생성
# 2. 특정 문서에 대한 피쳐 생성 함수 정의
# 3. 긍정과 부정으로 분류해 놓은 영화 리뷰 가져오기
# 4. 문서 전체를 피쳐 셋으로 구성하고 학습과 검사로 나누어서 정확도 예측
# 5. 최빈값 2000개를 사용할 수 있도록 코드 추가 (show_accuracy_multi 함수)

# -------------------------------------------------------------------------------------- #

import nltk
from nltk.corpus import movie_reviews
import random


# [1번] 리뷰에 포함된 모든 단어의 빈도 분포를 계산해서 빈도가 높은 단어 피쳐 생성
def make_vocab():
    print('---------------- make_vocab()')

    # nltk.download('movie_reviews')

    # 어떤 형태로 문서를 섞건 빈도는 항상 동일하다.
    # 앞에서 documents를 섞고 있지만, 이 코드에는 영향을 주지 않는다.
    words = movie_reviews.words()
    print(len(words), words[:10])
    # 1583820, ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', 'a', 'church', 'party']

    # 100개 단위로 성능 시각화 진행하면 좋겠다.
    all_words = nltk.FreqDist(w.lower() for w in words)
    most_2000 = all_words.most_common(2000)
    print(most_2000[:5])
    # [(',', 77717), ('the', 76529), ('.', 65876), ('a', 38106), ('and', 35576)]

    # 원본 문서에 있는 아래 코드보다 훨씬 잘 나온다. 최빈값 2000이 훨씬 좋다.
    vocab_2000 = [w for w, _ in most_2000]

    # 이 코드는 순서 없이 2000개의 키를 가져온다. 빈도를 무시하기 때문에 좋은 결과를 볼 수 없다. (??)
    # 그러나, 키를 가져오기 때문에 중복된 단어는 존재하지 않는다.
    # vocab_2000 = list(all_words)[:2000]
    # 200(0.61), 500(0.68), 1000(0.75), 2000(0.78), 5000(0.80)

    # 리스트로 형변환을 하는 코드는 keys 함수 호출과 같다.
    # 실제 코드 구현에서는 의미가 분명한 keys 함수를 사용하는 것이 좋다.
    # print(vocab_2000[:5])            # ['plot', ':', 'two', 'teen', 'couples']
    # print(list(all_words.keys())[:5])   # ['plot', ':', 'two', 'teen', 'couples']

    # 딕셔너리와 동일하게 keys(), values(), items()를 갖고 있다. (앞에서 설명했음)
    # print(list(all_words.values())[:5]) # [1513, 3042, 1911, 151, 27]
    # print(list(all_words.items())[:5])  # [('plot', 1513), (':', 3042), ('two', 1911), ('teen', 151), ('couples', 27)]
    #
    # 'plot'을 1513번 출력하고, ':'을 3042번 출력.
    # print(list(all_words.elements()))

    return vocab_2000


# [2] 특정 문서에 대한 피쳐 생성 함수 정의
# doc에 포함된 모든 단어에 대해 단어 피쳐에 존재하는지의 결과 반환
def document_features(doc, vocab):
    doc_words, features = set(doc), {}  # doc은 매번 달라지는 파일이라서 이 안에서만 set을 호출할 수 있다.
    for word in vocab:
        # True or False. 특정 단어의 "있음/없음"을 표시.
        key = 'contains({})'.format(word)
        features[key] = (word in doc_words)

        # (결론은 실패. 오히려 좋지 않게 나온다. 확실한 것은 훨씬 오랜 시간이 걸린다. 원인은 파악하지 않았다.)
        # features[key] = doc.count(word)
    return features


# document_features 함수 동작 확인. 긍정(pos)으로 분류된 파일을 샘플로 사용.
def test_doc_features():
    vocab = make_vocab()

    cv957 = movie_reviews.words('pos/cv957_8737.txt')
    id957 = document_features(cv957, vocab)

    print('---------------- test_doc_features()')
    print(type(id957), len(id957))  # <class 'dict'> 2000 (word_features 갯수)

    print(cv957)  # ['capsule', ':', 'the', 'best', 'place', 'to', 'start', ...]
    print(id957)  # {'contains(plot)': True, 'contains(:)': True, ..., 'contains(folks)': False}

    # print(movie_reviews.raw('neg/cv957_8737.txt'))
    # capsule : the best place to start if you're a jackie chan newcomer .
    # roars along , never stops for breath , and frequently hilarious .
    # to talk about jackie chan as a " stuntman " is to miss a million things .
    # ... 생략


# [3번] 긍정과 부정으로 분류해 놓은 영화 리뷰 가져오기
def make_documents():
    print('---------------- make_documents()')
    categories = movie_reviews.categories()
    print(categories)  # ['neg', 'pos']

    neg = movie_reviews.fileids('neg')
    pos = movie_reviews.fileids('pos')

    print(len(neg), len(pos))  # 1000 1000

    print(neg)  # ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt', ...]
    print(pos)  # ['pos/cv000_29590.txt', 'pos/cv001_18431.txt', 'pos/cv002_15918.txt', ...]

    # 문서별로 긍정인지 부정인지의 결과를 함께 저장.
    categories_neg = [(list(movie_reviews.words(fileid)), 'neg') for fileid in neg]
    categories_pos = [(list(movie_reviews.words(fileid)), 'pos') for fileid in pos]

    documents = categories_neg + categories_pos
    return documents


# [4] 문서 전체를 피쳐 셋으로 구성하고 학습과 검사로 나누어서 정확도 예측
# 전체 문서에 포함된 각각에 대해 문서 피쳐 데이터셋 구성 및 학습, 검증
def show_accuracy():
    vocab = make_vocab()
    documents = make_documents()

    # random.seed(1)
    random.shuffle(documents)

    feature_sets = [(document_features(doc, vocab), c) for (doc, c) in documents]
    train_set, test_set = feature_sets[100:], feature_sets[:100]

    clf = nltk.NaiveBayesClassifier.train(train_set)

    # print('---------------- show_accuracy()')
    print('acc :', nltk.classify.accuracy(clf, test_set))  # acc : 0.78

    # clf.show_most_informative_features(5)
    # # Most Informative Features
    # #     contains(schumacher) = True              neg : pos    =     12.3 : 1.0
    # #         contains(welles) = True              neg : pos    =      8.3 : 1.0
    # #  contains(unimaginative) = True              neg : pos    =      7.7 : 1.0
    # #      contains(atrocious) = True              neg : pos    =      7.0 : 1.0
    # #           contains(mena) = True              neg : pos    =      7.0 : 1.0


def show_accuracy_multi(seeds):
    vocab = make_vocab()
    documents = make_documents()

    for i in range(len(seeds)):
        random.seed(1)
        random.shuffle(documents)

        feature_sets = [(document_features(doc, vocab), c) for (doc, c) in documents]
        train_set, test_set = feature_sets[100:], feature_sets[:100]

        clf = nltk.NaiveBayesClassifier.train(train_set)
        print('acc :', nltk.classify.accuracy(clf, test_set))


# test_doc_features()
show_accuracy()
# show_accuracy_multi([1, 5, 7, 9, 13])

# [1] 단어 2000개 단순 추출
# acc : 0.75
# acc : 0.79
# acc : 0.82
# acc : 0.75
# acc : 0.87

# [2] 빈도에 따른 추출
# acc : 0.82
# acc : 0.79
# acc : 0.84
# acc : 0.83
# acc : 0.9

# neg와 pos 각각을 대상으로 FreqDist 변수를 만들어서
# 앞의 5개 단어에 각각의 빈도를 조사해 보면 neg인지 pos인지 나오지만, 값이 정확히 일치하지는 않는다.
# 실제 값은 naive bayes 확률 이론에 맞게 계산해야 한다.
def test():
    neg = movie_reviews.fileids('neg')
    pos = movie_reviews.fileids('pos')

    categories_neg = [word.lower() for fileid in neg for word in movie_reviews.words(fileid)]
    categories_pos = [word.lower() for fileid in pos for word in movie_reviews.words(fileid)]

    freq_neg = nltk.FreqDist(categories_neg)
    freq_pos = nltk.FreqDist(categories_pos)

    print(freq_neg['schumacher'], freq_pos['schumacher'])  # 11 2
    print(freq_neg['unimaginative'], freq_pos['unimaginative'])  # 12 1
    print(freq_neg['mena'], freq_pos['mena'])  # 54 3


# test()
