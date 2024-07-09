#!/usr/bin/env python

import faiss
import os,sys,argparse,time
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

t0 = time.time()

print('verify_normalize_embedding.py: ' + str(sys.argv), file=sys.stderr)
sys.stderr.flush()

# assumes the input directory contain 
#   embedding.f  sequence of N by K floats32
#   map.old_to_new.i  sequence of N int32
#   record_size  specifies K

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_directory", help="a directory", required=True)
parser.add_argument("--batch_size", type=int, default=1000)
args = parser.parse_args()

def record_size_from_dir(dir):
    with open(dir + '/record_size', 'r') as fd:
        return int(fd.read().split()[0])

def map_from_dir(dir):
    fn = dir + '/map.old_to_new.i'
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.int32, shape=(int(fn_len/4)), mode='r')

def embedding_from_dir(dir, K):
    fn = dir + '/embedding.norm.f'
    fn_len = os.path.getsize(fn)
    return np.memmap(fn, dtype=np.float32, shape=(int(fn_len/(4*K)), K), mode='r')

def directory_to_config(dir):
    K = record_size_from_dir(dir)
    return { 'record_size' : K,
             'dir' : dir,
             'map' : map_from_dir(dir),
             'embedding' : embedding_from_dir(dir, K)}

config = directory_to_config(args.input_directory)
embedding = config['embedding']

batches = np.arange(0, embedding.shape[0], args.batch_size)

for start,end in zip(batches, batches[1:]):
    embedding_norm = np.linalg.norm(embedding[start:end,:], axis=1) # should be unit length
    print('start = %d, embedding norm: %f +- %f, X.shape: %s; should be unit length with almost no variance' % (start, np.mean(embedding_norm), np.sqrt(np.var(embedding_norm)), str(embedding.shape)))

print('%0.f sec: done' % (time.time() -t0), file=sys.stderr)



