#Day_03_04_word2vec_basic.py

def extract(token_count, target, window_size):
    start = max(target - window_size, 0)#음수 값이 나오면 안되니깐
    end = min(target + window_size, token_count -1) + 1#범위 밖으로 나가면 안되니깐

    # [3, 4, 5, 6, 7] : 5 -> [3, 4, 5, 7]
    return [i for i in range(start, end) if i != target]

def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for y in range(token_count):
        x = extract(token_count, y, window_size)
        #print(y, x)

        if is_skipgram:
            print(*['({}, {})'.format(tokens[y], tokens[i]) for i in x])
        else:
            print('({}, {})'.format([tokens[i] for i in x], tokens[y]))
    print()

tokens = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']


# window_size + center + window_size = context_size
#show_dataset(tokens, 1, True)
#show_dataset(tokens, 1, False)


show_dataset(tokens, 2, True)
# (the, quick) (the, brown)
# (quick, the) (quick, brown) (quick, fox)
# (brown, the) (brown, quick) (brown, fox) (brown, jumped)
# (fox, quick) (fox, brown) (fox, jumped) (fox, over)
# (jumped, brown) (jumped, fox) (jumped, over) (jumped, the)
# (over, fox) (over, jumped) (over, the) (over, lazy)
# (the, jumped) (the, over) (the, lazy) (the, dog)
# (lazy, over) (lazy, the) (lazy, dog)
# (dog, the) (dog, lazy)
show_dataset(tokens, 2, False)
# (['quick', 'brown'], the)
# (['the', 'brown', 'fox'], quick)
# (['the', 'quick', 'fox', 'jumped'], brown)
# (['quick', 'brown', 'jumped', 'over'], fox)
# (['brown', 'fox', 'over', 'the'], jumped)
# (['fox', 'jumped', 'the', 'lazy'], over)
# (['jumped', 'over', 'lazy', 'dog'], the)
# (['over', 'the', 'dog'], lazy)
# (['the', 'lazy'], dog)