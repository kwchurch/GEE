#!/usr/bin/env python

import sys,os,argparse,scipy,psutil
import numpy as np
import copy
from numpy import linalg as LA
from sklearn.metrics.cluster import adjusted_rand_score
import faiss
from sklearn import metrics
import time
# node2vec
# from node2vec import Node2Vec
import networkx as nx
# for sparse matrix
from scipy import sparse
#early stop
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

print('GraphEncoder_clustering: ' + str(sys.argv), file=sys.stderr)

GEEsrc=os.environ.get('GEEsrc')

kwc_save_offset=0

t0 = time.time()                # added by kwc

# import numpy as np
# import argparse
# from sklearn.metrics.pairwise import cosine_similarity

# apikey=os.environ.get('SPECTER_API_KEY')

parser = argparse.ArgumentParser()
parser.add_argument("-O", "--output", help="output file", required=True)
parser.add_argument("--save_prefix", help="output file", default=None)
parser.add_argument("-G", "--input_graph", help="input graph (readable by scipy.sparse.load_npz)", default=None)
parser.add_argument("-d", "--input_directory", help="input directory with embedding", default=None)
parser.add_argument("-K", "--n_components", type=int, help="hidden dimensions [defaults = 32]", default=32)
parser.add_argument("--Laplacian", type=int, help="Laplacian [defaults = 1 (True)]", default=1)
parser.add_argument("--MaxIter", type=int, help="MaxIter [defaults = 50]", default=50)
parser.add_argument("--safe_mode", type=int, help="set to nonzero to be super careful", default=0)
args = parser.parse_args()

# Supress/hide the warning
# invalide devide resutls will be nan
np.seterr(divide='ignore', invalid='ignore')

def map_int64(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int64, shape=(int(fn_len/8)), mode='r')

def map_int32(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

def map_float32(fn):
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.float32, shape=(int(fn_len/4)), mode='r')

def my_int(s):
    for i,c in enumerate(s):
        if not c.isdigit():
            return int(s[0:i])

def record_size_from_dir(dir):
    with open(dir + '/record_size', 'r') as fd:
        return my_int(fd.read())

def map32_from_dir(dir):
    fn = dir + '/map.old_to_new.i'
    if not os.path.exists(fn): return None
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

def imap32_from_dir(dir):
    fn = dir + '/map.new_to_old.i'
    if not os.path.exists(fn): return None
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

def map64_from_dir(dir):
    fn = dir + '/map.old_to_new.sorted.L'
    if not os.path.exists(fn): return None
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=int, shape=(int(fn_len/8)), mode='r')

def embedding_from_dir(dir, K):
    fn = dir + '/embedding.f'
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.float32, shape=(int(fn_len/(4*K)), K), mode='r')

def directory_to_config(dir):
  if dir is None: return
  K = record_size_from_dir(dir)
  return { 'record_size' : K,
           'dir' : dir,
           'map32' : map32_from_dir(dir),
           'imap32' : imap32_from_dir(dir),
           'map64' : map64_from_dir(dir),
           'embedding' : embedding_from_dir(dir, K)}

def read_graph(fn):
  if fn is None: return
  return { 'X0' : map_int32(fn),
           'X1' : map_int32(fn),
           'X2' : map_float32(fn),
           }

kwc_save_offset=0

def save_X(G):
  global kwc_save_offset
  assert not args.save_prefix is None, '--save_prefix must be specified'

  if args.safe_mode > 0:
    X0path = args.save_prefix + '.X0.%d.i' % kwc_save_offset
    X1path = args.save_prefix + '.X1.%d.i' % kwc_save_offset
    X2path = args.save_prefix + '.X2.%d.f' % kwc_save_offset
    if os.path.exists(X0path):
      return X0path,X1path,X2path
  else:
    X0path = args.save_prefix + '.X0.i'
    X1path = args.save_prefix + '.X1.i'
    X2path = args.save_prefix + '.X2.f'

  X0 = G['X0'].astype(np.int32)
  X1 = G['X1'].astype(np.int32)
  X2 = G['X2'].astype(np.float32)

  X0.tofile(X0path)
  X1.tofile(X1path)
  X2.tofile(X2path)
  return X0path,X1path,X2path

# G = (V, E)
# K is the number of hidden dimensions: args.n_components
# Z is an embedding with shape (|V|, K)
# X0, X1, X2 specify the graph
#    all three vectors have the same length: s = |E|;
#    The elements in X0, X1 specify the edges E
#    X2 specifies the weights for each edge

def create_Z(G, Y, n):
  """
  X0, X1, X2 specifies the graph: (V, E)
  all three vectors have the same length: s = |E|;
  The elements in X0, X1 specify the edges E
  X2 specifies the weights for each edge
  """

  global kwc_save_offset

  kwc_gee_t0 = time.time()          # added by kwc
  
  X0path,X1path,X2path = save_X(G)

  # to simplify things, lets avoid more complicated cases
  if Y.shape[1] != 1: return None

  YY = Y.reshape(-1).astype(np.int32)
  Ypath = args.save_prefix + '.Y.%d.i' % kwc_save_offset
  Zpath = args.save_prefix + '.Z_kwc.%d.f' % kwc_save_offset
  YY.tofile(Ypath)

  cmd = '%s/C/GEE %s %s %s %s > %s' % (JSALTsrc, X0path, X1path, X2path, Ypath, Zpath)
  print(cmd, file=sys.stderr)
  os.system(cmd)
  assert os.path.exists(Zpath), 'cmd failed'    
  Z = np.memmap(Zpath, dtype=np.float32, shape=(n, k), mode='r')
  print('gee kwc (plus time for saving): %0.3f; time so far: %0.3f; memory = %0.2f GBs' % (time.time() - kwc_gee_t0, time.time() - t0, psutil.Process().memory_info().rss / 1e9), file=sys.stderr) # added by kwc
  print(psutil.virtual_memory(), file=sys.stderr)
  sys.stderr.flush()
  return Z

def faiss_kmeans(Z, K, max_iter):
  kmeans = faiss.Kmeans(d=Z.shape[1], k=K, niter=max_iter)
  kmeans.train(Z)
  dist, labels = kmeans.index.search(queries, 1)
  labels = labels.reshape(-1)
  return labels

def create_Y(Z):
  kmeans_t0 = time.time()
  K = args.n_components
  max_iter = args.MaxIter
  # kmeans = MiniBatchKMeans(n_clusters=K, max_iter = max_iter).fit(Z)
  # labels = kmeans.labels_ # shape(n,)
  labels = faiss_kmeans(Z, K, max_iter)
  printf('%0.3f sec: kmeans' % (time.time() - kmeans_t0))
  print(psutil.virtual_memory(), file=sys.stderr)
  sys.stderr.flush()
  return labels

assert args.input_directory or args.graph, 'need to specify --input_directory or --input_graph'
 
config = directory_to_config(args.input_directory)
G = read_graph(args.input_graph)

if not config is None:
  Z1 = config['embedding']
  Y1 = create_Y(Z)
  Z2 = create_Z(Y)


print('done, %0.3f sec' % (time.time() - t0), file=sys.stderr)  # added by kwc
