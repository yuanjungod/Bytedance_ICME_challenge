print("okokokQQQQQQQ!!!")
import time


def gene_test():
    count = 0
    while True:
        yield count
        count += 1


gaga = [[i for i in range(10000)] for _ in range(1)]
gen = gene_test()


for i in range(10000):
    time.sleep(0.1)
    print(i)
