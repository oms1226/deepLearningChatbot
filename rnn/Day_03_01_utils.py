#Day_03_01_utils.py


def twice(n):
    return n * 2

def proxy(func, n):
    print(func)
    print(func(n))


f = twice
print(f)
print(twice)

print(twice(3))
print(f(3))

#proxy(3)
proxy(twice, 3)