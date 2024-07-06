#!/usr/bin/env python

import sys,os,argparse,scipy,psutil,subprocess
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import faiss
from sklearn import metrics
import time

print('GEE.py: ' + str(sys.argv), file=sys.stderr)
sys.stderr.flush()

GEEsrc=os.environ.get('GEEsrc')

save_offset=0

t0 = time.time()

parser = argparse.ArgumentParser()
# parser.add_argument("-O", "--output", help="output file", required=True)
parser.add_argument("--save_prefix", help="output file", default=None)
parser.add_argument("-G", "--input_graph", help="input graph (pathname minus .X.i)", default=None)
parser.add_argument("-d", "--input_directory", help="input directory with embedding", default=None)
parser.add_argument("-K", "--hidden_dimensions", type=int, help="override dimensions in embedding (for upsampling)", default=None)
parser.add_argument("--seed", type=int, help="set random seed", default=None)
parser.add_argument("--brain_damage", type=int, help="set <arg> rows of Z to zero", default=None)
parser.add_argument("--Laplacian", type=int, help="Laplacian [defaults = 1 (True)]", default=1)
parser.add_argument("--MaxIter", type=int, help="MaxIter [defaults = 50]", default=50)
parser.add_argument("--safe_mode", type=int, help="set to nonzero to be super careful", default=0)
parser.add_argument("--iteration_start", type=int, help="options to restart with iteration", default=0)
parser.add_argument("--iteration_end", type=int, help="options to restart with iteration", default=20)
args = parser.parse_args()

# Supress/hide the warning
# invalid results from division will be nan
np.seterr(divide='ignore', invalid='ignore')

if not args.seed is None:
    np.random.seed(args.seed)

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

def read_graph(fn, old_to_new):
  if fn is None: return
  G =  { 'X0' : old_to_new[map_int32(fn + '.X.i')],
         'X1' : old_to_new[map_int32(fn + '.Y.i')]}
  X2path = fn + '.W.f'
  if not os.path.exists(X2path):
    G['X2'] = np.ones(len(G['X0']), dtype=np.float32)
  else:
    G['X2'] = map_float32(X2path)
  return G

save_offset=args.iteration_start

def save_X(G):
  global save_offset
  assert not args.save_prefix is None, '--save_prefix must be specified'

  if args.safe_mode > 0:
    X0path = args.save_prefix + '.X0.%d.i' % save_offset
    X1path = args.save_prefix + '.X1.%d.i' % save_offset
    X2path = args.save_prefix + '.X2.%d.f' % save_offset
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

def create_Z(G, Y, Zprev_path, norm):
  """
  X0, X1, X2 specifies the graph: (V, E)
  all three vectors have the same length: s = |E|;
  The elements in X0, X1 specify the edges E
  X2 specifies the weights for each edge
  """

  n = len(Y)
  global save_offset

  gee_t0 = time.time()          # added by kwc
  
  X0path,X1path,X2path = save_X(G)

  # to simplify things, lets avoid more complicated cases
  # if Y.shape[1] != 1: return None

  YY = Y.reshape(-1).astype(np.int32)
  Ypath = args.save_prefix + '.Y.%d.i' % save_offset
  Zpath = args.save_prefix + '.Z.%d.f' % save_offset
  YY.tofile(Ypath)

  # cmd = GEEsrc + '/C/GEE ' + ' '.join([X0path, X1path, X2path, Ypath, Zprev_path, Zpath + '.err', Zpath])
  cmd = GEEsrc + '/C/GEE2 ' + ' '.join([
      "--err", Zpath + '.err',
      "--X0", X0path,
      "--X1", X1path, 
      "--X2", X2path,
      "--Y", Ypath, 
      "--Zprev", Zprev_path, 
      "--Zout", Zpath])

  if norm: cmd += ' --normalize'
  print('cmd: ' + cmd, file=sys.stderr)
  sys.stderr.flush()
  # os.system(cmd)
  result = subprocess.run(cmd, capture_output=True, shell=True)
  print('return code from cmd: %d, stdout: %s, stderr: %s]' % (result.returncode, result.stdout.decode('utf-8'), result.stderr.decode('utf-8')), file=sys.stderr)

  assert os.path.exists(Zpath), 'cmd failed'    
  # Z = np.memmap(Zpath, dtype=np.float32, shape=(n, k), mode='r')
  Z = map_float32(Zpath).reshape(n, -1)
  print('gee (plus time for saving): %0.3f; time so far: %0.3f; memory = %0.2f GBs' % (time.time() - gee_t0, time.time() - t0, psutil.Process().memory_info().rss / 1e9), file=sys.stderr) # added by kwc
  print(psutil.virtual_memory(), file=sys.stderr)
  sys.stderr.flush()
  return Z,Zpath

def faiss_kmeans(Z, K, max_iter):
    kwargs = {}
    if not args.seed is None:
        kwargs['seed'] = args.seed
    kmeans = faiss.Kmeans(d=Z.shape[1], k=K, niter=max_iter)
    kmeans.train(Z)
    print('%0.3f sec: faiss_kmeans, finished training' % (time.time() - t0), file=sys.stderr)
    dist, labels = kmeans.index.search(Z, 1)
    print('%0.3f sec: faiss_kmeans, found labels, RMS = %f' % (time.time() - t0, np.sqrt(np.mean(dist*dist))), file=sys.stderr)
    return labels.reshape(-1)

def create_Y(Z, K):
    # K = args.n_components
    max_iter = args.MaxIter
    # kmeans = MiniBatchKMeans(n_clusters=K, max_iter = max_iter).fit(Z)
    # labels = kmeans.labels_ # shape(n,)
    labels = faiss_kmeans(Z, K, max_iter)
    return labels

assert args.input_directory or args.graph, 'need to specify --input_directory or --input_graph'
 
config = directory_to_config(args.input_directory)
G = read_graph(args.input_graph, config['map32'])

if args.iteration_start == 0:
    if not config is None:
        Z1 = config['embedding']
        Zpath = args.input_directory + '/embedding.f'
else:
    Zpath = args.save_prefix + '.Z.%d.f' % (save_offset-1)
    Z1 =  map_float32(Zpath).reshape(-1, config['record_size'])

Yprev = None

# Create extra col with zeros
# for upsampling
if args.iteration_start == 0 and not args.hidden_dimensions is None:
    if args.hidden_dimensions <= Z1.shape[1]:
        newZ1 = np.copy(Z1[:,0:args.dimensions])
    else:
        newZ1 = np.zeros((Z1.shape[0], args.hidden_dimensions), dtype=np.float32)
        newZ1[:,0:Z1.shape[1]] = Z1
    Z1 = newZ1
    Zpath = args.save_prefix + '.Z.init.f'
    Z1.tofile(Zpath)

# Create rows with zeros
if args.iteration_start == 0 and not args.brain_damage is None:
    newZ1 = np.copy(Z1[:,0:args.dimensions])
    newZ1[np.random.choice(Z1.shape[0], args.brain_damage),:] = 0
    Z1 = newZ1
    Zpath = args.save_prefix + '.Z.init.f'
    Z1.tofile(Zpath)
    
for iteration in range(args.iteration_start, args.iteration_end):

  print('%0.3f sec: working on iteration: %d' % (time.time() - t0, iteration), file=sys.stderr)
  sys.stderr.flush()

  Y1 = create_Y(Z1, Z1.shape[1])

  print('%0.3f sec: Y1 computed, iteration: %d' % (time.time() - t0, iteration), file=sys.stderr)
  if not Yprev is None:
      score = adjusted_rand_score(Y1, Yprev)
      print('%0.3f sec: iteration %d, ARI score %f' % (time.time() - t0, iteration, score), file=sys.stderr)
      if score > 0.999: break

  print(psutil.virtual_memory(), file=sys.stderr)
  sys.stderr.flush()

  Z2,newZpath = create_Z(G, Y1, Zpath, iteration==0)

  print('%0.3f sec: Z2 computed, iteration: %d' % (time.time() - t0, iteration), file=sys.stderr)
  print(psutil.virtual_memory(), file=sys.stderr)
  sys.stderr.flush()

  Z1 = Z2
  Zpath = newZpath
  Yprev = Y1
  save_offset += 1

print('done, %0.3f sec' % (time.time() - t0), file=sys.stderr)  # added by kwc
