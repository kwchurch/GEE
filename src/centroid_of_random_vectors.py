#!/usr/bin/env python

import numpy as np
import time,os,sys,argparse
from sklearn.preprocessing import normalize

print('centroid_of_random_vectors.py: ' + str(sys.argv), file=sys.stderr)
sys.stderr.flush()

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n", type=int, help="number of random vectors to pick", required=True)
parser.add_argument("-d", "--dimensions", type=int, help="dimensions", required=True)
args = parser.parse_args()

X = normalize(np.random.random((args.n, args.dimensions)))
Xlen = np.linalg.norm(X, axis=1) # should be unit length
centroid = np.mean(X, axis=0)

print('centroid: norm = %f, shape = %s; should be less than unit length, with shape = (d=%d,)' % (np.linalg.norm(centroid), str(centroid.shape), args.dimensions))

print('length of X: %f +- %f, X.shape: %s; should be unit length with almost no variance, with shape = (n=%d,)' % (np.mean(Xlen), np.sqrt(np.var(Xlen)), str(Xlen.shape), args.n))

# print('random 10 values of X: ' + str(X[np.random.choice(len(X), 10),:]))
