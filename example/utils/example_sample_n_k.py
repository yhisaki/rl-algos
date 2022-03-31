import random
import time

import numpy as np

from rl_algos.utils.sample_n_k import sample_n_k

if __name__ == "__main__":

    N = 10000000
    K = 10

    # sample_n_k benchmark
    start = time.time()
    sampled = sample_n_k(N, K)
    elapsed_time = time.time() - start
    print(f"sample_n_k      : {elapsed_time}")

    # np.random.choice benchmark
    start = time.time()
    sampled = np.random.choice(N, K, replace=False)
    elapsed_time = time.time() - start
    print(f"np.random.choice: {elapsed_time}")

    # random.sample benchmark
    start = time.time()
    sampled = random.sample(range(N), K)
    elapsed_time = time.time() - start
    print(f"random.sample   : {elapsed_time}")
