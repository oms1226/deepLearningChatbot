# Day_03_04_word2vec_basic.py


def extract(token_count, target, window_size):
    start = max(target - window_size, 0)
    end = min(target + window_size, token_count - 1) + 1

    # [3, 4, 5, 6, 7] : 5 -> [3, 4, 6, 7]
    return [i for i in range(start, end) if i != target]


def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for y in range(token_count):
        x = extract(token_count, y, window_size)
        # print(y, x)

        if is_skipgram:
            print(*['({}, {})'.format(tokens[y], tokens[i]) for i in x])
        else:
            print('({}, {})'.format([tokens[i] for i in x], tokens[y]))
    print()



tokens = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']

# window_size + center + window_size : context_size
# show_dataset(tokens, 1, True)
# show_dataset(tokens, 1, False)

show_dataset(tokens, 2, True)
show_dataset(tokens, 2, False)









