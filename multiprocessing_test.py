# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:59:01 2016

@author: eejvt
"""

import multiprocessing

import time
import numpy as np

def worker(num):
    """thread worker function"""
    np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3*num
    return time.time()

iters=2

print '------------'
start = time.time()
if __name__ == '__main__':
    #jobs = []
    nums=[3*i for i in range(iters)]
    for num in nums:
        #p = multiprocessing.Process(target=worker, args=(num,))
        #jobs.append(p)
        worker(num)
        #p.start()
    
end = time.time()
print 'usual'
print(end - start)



print '------------'



result_queue = multiprocessing.Queue()
start = time.time()
if __name__ == '__main__':
    jobs = []
    nums=[3*i for i in range(iters)]
    for num in nums:
        p = multiprocessing.Process(target=worker, args=(num,))
        jobs.append(p)
        p.start()
        print jobs
for proc in jobs: proc.join()
results = [result_queue.get() for mc in jobs]
end = time.time()
print 'multiprocessing'
print(end - start)



#%%
import multiprocessing
from os import getpid

def worker(procnum):
    print 'I am number %d in process %d' % (procnum, getpid())
    return getpid()

def worker(num):
    """thread worker function"""
    np.load('/nfs/a201/eejvt//MARINE_PARAMETERIZATION/FOURTH_TRY/INP_marine_alltemps.npy')*1e-3*num
    return time.time()

start = time.time()
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 32)
    a=pool.map(worker, range(5))
end = time.time()
print 'multiprocessing'
print(end - start)


