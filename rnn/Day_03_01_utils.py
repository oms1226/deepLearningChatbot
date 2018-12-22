#Day_03_01_utils.py


def twice(n):
    return n * 2

def proxy(func, n):
    print(func)
    print(func(n))

lamb = lambda n: n * 2 ##return 이 생략된 한줄짜리 함수

f = twice
print(f)
print(twice)

print(twice(3))
print(f(3))

#proxy(3)
proxy(twice, 3)

print(lamb(7))
print((lambda n: n * 2)(7))

proxy(lambda n: n * 2, 3)