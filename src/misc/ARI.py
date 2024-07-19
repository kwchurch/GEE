#!/usr/bin/env python

import numpy as np
import sys,os
from sklearn.metrics.cluster import adjusted_rand_score

def map_int32(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

assert len(sys.argv) >= 3, 'usage: ARI.py files'

for i in range(2, len(sys.argv)):
    Y0 = map_int32(sys.argv[i-1])
    Y1 = map_int32(sys.argv[i])
    print(str(adjusted_rand_score(Y0, Y1)) + '\t' + sys.argv[i-1] + '\t' + sys.argv[i])
    sys.stdout.flush()


