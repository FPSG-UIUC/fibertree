import random
import timeit

from fibertree import Fiber, Tensor

def setup(seed):
    K = 100
    M = 100

    A_KM = Tensor.fromRandom(rank_ids=["K", "M"], shape=[K, M], density=[0.8, 0.3], seed=seed)
    B_K = Tensor.fromRandom(rank_ids=["K"], shape=[K], density=[0.7], seed=seed + 1000)

    return (A_KM.getRoot(), B_K.getRoot())

def spmv(a_k, b_k):
    z_m = Fiber()
    for k, (a_m, b_val) in a_k & b_k:
        for m, (z_ref, a_val) in z_m << a_m:
            z_ref += a_val * b_val

if __name__ == '__main__':
    num = 100
    print(timeit.timeit("spmv(a_k, b_k)", globals=locals(), setup="a_k, b_k = setup(random.random())", number=num) / num)
