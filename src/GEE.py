#!/usr/bin/env python

import sys,os,argparse,scipy,psutil,subprocess
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import faiss
from sklearn import metrics
import time
from scipy.sparse import load_npz, csr_matrix

print('GEE.py: ' + str(sys.argv), file=sys.stderr)
sys.stderr.flush()

GEEsrc=os.environ.get('GEEsrc')

save_offset=0

t0 = time.time()

# NOTATION:
# Let G=(V,E) be a graph
# and Z be an embedding with K hidden dimensions
# Thus, Z is an array with dtype of np.float32 and shape: (|V|, K)
# Y is a sequence of |V| class labels, where 0 <= Y[i] < K

# G will be represented with X0, X1, X2.  All three vectors have length of |E|.
# Edges go from x0 in X0 to x1 in X2 with weights x3 in X3.
# X0 and X1 are stored with dtype of np.int32, and X2 is stored with dtype of np.float32
# X2 is optional, and defaults to a vector of ones (if not specified).

# INITIALIZATION:
# If --cold_start is specified, then Z is initialized to zeros, and Y is initialized with random labels (maintaining the invariant, 0 <= Y[i] < K)
# If --new_cold_start is specified, then initialization is similar to --cold_start, but with an attempt to assign the same label to Y if the vertices are near one another in G.
# If not, Z is initialized with values from --input_directory
#
# Experimental results suggest that --new_cold_start is better than --cold_start, but it is even better to start from ProNE
# By better, we can measure cosine similarities of papers that are near one another in citation graph,
# as well as intrinsic measures such as RMS distances of vectors to their nearest centroid (in kmeans), and ARI similarities of Yprev to Ynext.

# ITERATIONS:
# For each iteration,
#   we estimate then next Y from the previous Z
#   and then we use that Y to estimate the next Z

# RESTARTING:
# By default, we start with --iteration_start of 0 and --iteration_end of 20
# Intermediate values are written to --save_prefix, so one can start with a previously completed iteration
# and continue with additional iterations if desired.
# One can even continue with additional iterations beyond previous values of iteration_end.

# INCREMENTAL UPDATES:
#
# INCREMENTAL UPDATES Case 1: adding rows
# If one computed a previous embedding, Zprev, with a previous graph, Gprev=(Vprev, Eprev),
# one can use that embedding with a new graph, G=(V,E), where Vprev is a subset of V and Eprev is a subset of E.
# The new rows of Z are initialized with zeros.
#
# INCREMENTAL UPDATES Case 2: adding columns
# One can specify additional columns by using the optional arg, --hidden_dimensions, to specify more rows than Z.
# If so, the new columns of Z are initialized with zeros.

# BRAIN DAMAGE:
# The optional arg, --brain_damage, sets some rows of Z to zero.  Hopefully, the iteration will
# recover reasonable values in that case.

parser = argparse.ArgumentParser()
# parser.add_argument("-O", "--output", help="output file", required=True)
parser.add_argument("--cold_start", help="start with random classes for Y and zeros for Z", action='store_true')
parser.add_argument("--new_cold_start", help="like cold_start but try to assign vertices near one another the same label in Y", action='store_true')
parser.add_argument("--save_prefix", help="output file", default=None)
parser.add_argument("-G", "--input_graph", help="input graph (pathname minus .X.i or sparse graph in file ending with .npz)", required=True)
parser.add_argument("-d", "--input_directory", help="input directory with embedding", default=None)
parser.add_argument("-K", "--hidden_dimensions", type=int, help="defaults to embedding shape[1] if not specified, but can be overridden (for upsampling); must be specified if --input directory is not specified", default=None)
parser.add_argument("--seed", type=int, help="set random seed (if specified)", default=None)
parser.add_argument("--brain_damage", type=int, help="set <arg> rows of Z to zero", default=None)
# parser.add_argument("--Laplacian", type=int, help="Laplacian [defaults = 1 (True)]", default=1)
parser.add_argument("--MaxIter", type=int, help="MaxIter (used in kmeans) [defaults = 50]", default=50)
# parser.add_argument("--safe_mode", type=int, help="set to nonzero to be super careful", default=0)
parser.add_argument("--iteration_start", type=int, help="defaults to 0; specify with nonzero to restart at a previously completed iteration", default=0)
parser.add_argument("--iteration_end", type=int, help="defaults to 20; specify to stop eariler, or continue longer", default=20)
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
    if dir is None:
        assert not args.hidden_dimensions is None, 'need to specify --hidden_dimensions if --input_directory is not specified'
        return { 'record_size' : args.hidden_dimensions,
                 'dir' : None,
                 'map32' : None,
                 'imap32' : None,
                 'map64' : None,
                 'embedding' : None}
    K = record_size_from_dir(dir)
    return { 'record_size' : K,
             'dir' : dir,
             'map32' : map32_from_dir(dir),
             'imap32' : imap32_from_dir(dir),
             'map64' : map64_from_dir(dir),
             'embedding' : embedding_from_dir(dir, K)}

def read_graph(fn, old_to_new):
    """
    X0, X1, X2 specifies the graph: (V, E)
    all three vectors have the same length: s = |E|;
    The elements in X0, X1 specify the edges E
    X2 specifies the weights for each edge
    """

    if fn is None: return
    
    if fn.endswith('.npz'):
        M = load_npz(fn)
        X0,X1 = M.nonzero()
        G = { 'X0': X0, 'X1' : X1 }

    elif old_to_new is None:
        G =  { 'X0' : map_int32(fn + '.X.i'),
               'X1' : map_int32(fn + '.Y.i')}
    else:
        G =  { 'X0' : old_to_new[map_int32(fn + '.X.i')],
               'X1' : old_to_new[map_int32(fn + '.Y.i')]}

    assert len(G['X0']) == len(G['X1']), 'expected |X0| == |X1|'
    G['nVertices'] = 1 + max(np.max(G['X0']), np.max(G['X1']))
    G['nEdges'] = len(G['X0'])

    X2path = fn + '.W.f'
    if not os.path.exists(X2path):
        G['X2'] = np.ones(len(G['X0']), dtype=np.float32)
    else:
        G['X2'] = map_float32(X2path)
        assert len(G['X0']) == len(G['X2']), 'expected |X0| == |X2|'

    return G

save_offset=args.iteration_start

def save_X(G):
  global save_offset
  assert not args.save_prefix is None, '--save_prefix must be specified'
  
  X0path = args.save_prefix + '.X0.i'
  X1path = args.save_prefix + '.X1.i'
  X2path = args.save_prefix + '.X2.f'

  if args.iteration_start == 0:
      print('%0.3f sec: save_X, saving graph' % (time.time() - t0), file=sys.stderr)
      X0 = G['X0'].astype(np.int32)
      X1 = G['X1'].astype(np.int32)
      X2 = G['X2'].astype(np.float32)

      X0.tofile(X0path)
      X1.tofile(X1path)
      X2.tofile(X2path)

  return X0path,X1path,X2path

def create_Z(G, Y, Zprev_path, norm):
  n = len(Y)
  global save_offset

  gee_t0 = time.time()
  
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
      "--Zout", Zpath])

  if not Zprev_path is None:
      cmd += ' --Zprev ' + Zprev_path
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
    print('%0.3f sec: faiss_kmeans, finished training, stats: %s' % (time.time() - t0, '\n\t'.join(map(str, kmeans.iteration_stats))), file=sys.stderr)
    dist, labels = kmeans.index.search(Z, 1)
    print('%0.3f sec: faiss_kmeans, found labels, RMS = %f' % (time.time() - t0, np.sqrt(np.mean(dist*dist))), file=sys.stderr)
    return labels.reshape(-1)

def create_Y_from_cold_start(G):
    assert not args.hidden_dimensions is None, 'must specify --hidden_dimensions if --cold_start is specified'
    R = np.random.choice(args.hidden_dimensions, G['nVertices']).astype(np.int32)
    if not args.new_cold_start:
        return R
    else:
        # when possible, vertices near one another in G should receive the same label
        res = -np.ones(G['nVertices'], dtype=np.int32) # empty
        for x0,x1,rand in zip(G['X0'], G['X1'], R):
            if res[x0] < 0 and res[x1] < 0: res[x0] = res[x1] = rand
            elif res[x0] >= 0 and res[x1] >= 0: continue
            else: res[x0] = res[x1] = max(res[x0], res[x1])

        # There shouldn't be any values less than 0, but if there are, fill them in
        s = res < 0
        if np.sum(s) > 0: res[s] = R[s]

        return res

def create_Y(Z):
    # K = args.n_components
    K = Z.shape[1]
    max_iter = args.MaxIter
    # kmeans = MiniBatchKMeans(n_clusters=K, max_iter = max_iter).fit(Z)
    # labels = kmeans.labels_ # shape(n,)
    labels = faiss_kmeans(Z, K, max_iter)
    return labels
 
config = directory_to_config(args.input_directory)
if config is None:
    G = read_graph(args.input_graph, None)
else:
    G = read_graph(args.input_graph, config['map32'])

if args.iteration_start > 0:
    Zpath = args.save_prefix + '.Z.%d.f' % (save_offset-1)
    Z1 =  map_float32(Zpath).reshape(-1, config['record_size'])
elif args.cold_start or args.new_cold_start:
    Z1 = None
    Zpath = None
elif args.iteration_start == 0:
    Z1 = config['embedding']
    Zpath = args.input_directory + '/embedding.f'
else: assert False, 'there are no other cases'

Yprev = None

# Create extra col with zeros
# for upsampling
if not Z1 is None and args.iteration_start == 0 and not args.hidden_dimensions is None:
    if args.hidden_dimensions <= Z1.shape[1]:
        newZ1 = np.copy(Z1[:,0:args.hidden_dimensions])
    else:
        newZ1 = np.zeros((Z1.shape[0], args.hidden_dimensions), dtype=np.float32)
        newZ1[:,0:Z1.shape[1]] = Z1
    config['record_size'] = args.hidden_dimensions
    Z1 = newZ1
    Zpath = args.save_prefix + '.Z.init.f'
    Z1.tofile(Zpath)

# Replace rows with zeros
if Z1 is None and args.iteration_start == 0 and not args.brain_damage is None:
    newZ1 = np.copy(Z1[:,0:args.hidden_dimensions])
    newZ1[np.random.choice(Z1.shape[0], args.brain_damage),:] = 0
    Z1 = newZ1
    Zpath = args.save_prefix + '.Z.init.f'
    Z1.tofile(Zpath)
    
for iteration in range(args.iteration_start, args.iteration_end):

  print('%0.3f sec: working on iteration: %d' % (time.time() - t0, iteration), file=sys.stderr)
  sys.stderr.flush()

  if Z1 is None: Y1 = create_Y_from_cold_start(G)
  else: Y1 = create_Y(Z1)

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
