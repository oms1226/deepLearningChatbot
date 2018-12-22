# Day_02_04_rnn_4_1_comprehension.py
import random

# 컴프리헨션 : 집계 함수에 전달할 컬렉션을 만드는 한 줄짜리 반복문

for i in range(5):
    i

for i in range(5):
    if i % 2:
        i

[i for i in range(5)]
(i for i in range(5))
{i for i in range(5)}

print([i for i in range(5)])
print([0 for i in range(5)])
print(['yes' for i in range(5)])

a = [random.randrange(100) for _ in range(10)]
print(a)
print([i for i in a])

# 문제
# 1차원 리스트에서 홀수만 찾아보세요
print([i for i in a if i % 2])
print()

b1 = [random.randrange(100) for _ in range(10)]
b2 = [random.randrange(100) for _ in range(10)]
b3 = [random.randrange(100) for _ in range(10)]

b = [b1, b2, b3]

# 문제
# 2차원 리스트를 1차원으로 변환하세요
print([i for i in b])
print([0 for i in b])
print([[0] for i in b])

print([j for i in b for j in i])
print([j for i in b for j in i if j % 2])
print()

# 문제
# 2차원 리스트에서 홀수를 찾아주세요 (차원 유지)
print([[j for j in i if j % 2] for i in b])

print(sum(b1))

# 문제
# 2차원 리스트의 합계를 구하세요
print([sum(i) for i in b])
print(sum([sum(i) for i in b]))
print('-' * 50)

w = list('tensorflow')
print(w)
print(sorted('tensorflow'))
print(set(sorted('tensorflow')))

print({i for i in 'tensorflow'})
print(sorted({i for i in 'tensorflow'}))
print()

print([c for c in 'tensor'])
print([c for c in enumerate('tensor')])
print([(c, i) for i, c in enumerate('tensor')])
print()

d = {c: i for i, c in enumerate('tensor')}
print(d)

print(d['s'])







print('\n\n\n\n\n\n\n')