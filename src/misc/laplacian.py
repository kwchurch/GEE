#!/usr/bin/env python

import numpy as np
import time,os,sys,argparse

print('laplacian.py: ' + str(sys.argv), file=sys.stderr)
sys.stderr.flush()

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output pathname", required=True)
parser.add_argument("-G", "--input_graph", help="input graph (pathname minus .X.i)", default=None)
args = parser.parse_args()

def map_int32(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

def map_float32(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.float32, shape=(int(fn_len/4)), mode='r')


def laplacian_without_weights(X0, X1):
    D = np.zeros((len(X0),1))
    for v_i, v_j in zip(X0,X1):
        D[v_i] += 1
        if v_i != v_j:
          D[v_j] += 1

    D = np.power(D, -0.5)

    X2 = np.ones(len(X0))
    for i in range(len(X0)):
        X2[i] *= D[X0[i]] * D[X1[i]]

    return X2


def laplacian(X0, X1, X2):
    D = np.zeros((len(X0),1))
    for v_i, v_j, edg_i_j in zip(X0,X1,X2):
        D[v_i] += edg_i_j
        if v_i != v_j:
          D[v_j] += edg_i_j

    D = np.power(D, -0.5)

    X2 = np.copy(X2)
    for i in range(len(X0)):
        X2[i] *= D[X0[i]] * D[X1[i]]

    return X2


X0 = map_int32(args.input_graph + '.X0.i')
X1 = map_int32(args.input_graph + '.X1.i')
if os.path.exists(args.input_graph + '.X2.f'):
    X2 = map_float32(args.input_graph + '.X2.f')
    newX2 = laplacian(X0, X1, X2)
else:
    newX2 = laplacian_without_weights(X0, X1)

newX2.tofile(args.output)
