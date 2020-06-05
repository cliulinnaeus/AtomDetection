import multiprocessing as mp
from multiprocessing import pool
import numpy as np
import time

def my_func(x, y):
    # print(mp.current_process())
    return x*y
arr = np.linspace(0,10000,10001)

def main():
    pool = mp.Pool(mp.cpu_count())
    print(mp.cpu_count())
    pool.map(my_func, arr, arr)
    # result_set_2 = pool.map(my_func, np.linspace(0,9, 5))

    # print(result)
    # print(result_set_2)

def main_no_mp():
    for i in arr:
        my_func(i, i)

    


if __name__ == "__main__":
    prev_time = time.time()
    main()
    print(f"time1: {time.time() - prev_time}")
    
    prev_time = time.time()
    main_no_mp()
    print(f"time1: {time.time() - prev_time}")
