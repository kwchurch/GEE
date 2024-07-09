# Graph Encoder Embedding (GEE)

<h2>Overview</h2>

<a href="https://arxiv.org/pdf/2109.13098">GEE</a> is an alternative to 
<a href="https://www.researchgate.net/profile/Ming-Ding-2/publication/334844418_ProNE_Fast_and_Scalable_Network_Representation_Learning/links/5f1e97f292851cd5fa4b2285/ProNE-Fast-and-Scalable-Network-Representation-Learning.pdf">ProNE</a>.
Both methods input a graph, G, and
output an embedding, Z.  Cosines of rows of Z can be interpreted in
terms of random walks on G.

The bottleneck for ProNE is the SVD in the prefactorization step.  GEE
is faster than ProNE because it avoids the SVD, though we believe the GEE itertions are likely to get stuck in
suboptimal local minima, especially if we start from a cold start.  We
recommend starting GEE from a ProNE embedding for a simpler case with fewer rows
and fewer columns.

* fewer rows: start with ProNE based on a subgraph of G
* fewer columns: start with ProNE with fewer hidden dimensions

<h2>Notation</h2>

Let $G=(V,E)$ be a graph
and $Z$ be an embedding with $K$ hidden dimensions.
In other words, $Z$ is an array with dtype of np.float32 and shape: $(|V|, K)$.

Cosines of rows of $Z$ can be interpreted in terms of random walks on $G$.

$Y$ is a sequence of $|V|$ class labels, where $0 \le Y[i] < K$.  

**Note**: $K$ is both the number
of hidden dimensions in $Z$ as well as the number of class lables in $Y$.

$G$ will be represented with three vectors: $X0, X1, X2$.  All three vectors have length of $|E|$.
Edges go from $x0$ in $X0$ to $x1$ in $X2$ with weights $x3$ in $X3$.
$X0$ and $X1$ are stored with dtype of np.int32, and $X2$ is stored with dtype of np.float32
$X2$ is optional, and defaults to a vector of ones (if not specified).

$Z$ and $Y$ are both estimated with an iteration process that starts with initial values, $Z_0$ and $Y_0$,
and then estimates $Z_i$ and $Y_i$ on the $i^{th}$ iteration.  Each iteration may refer to values of $Y_{i-1}$ and $Z_{i-1}$
computed on the previous iteration.

Inputs to GEE:
* $G$ (required)
* options for initializing $Y_0$ and $Z_0$ (either from cold start or from something better)
* options for restarting iterations
* optional hyperparameters

Outputs:
* $Z_i$: embedding on the $i^{th}$ iteration
* $Y_i$: class labels on the $i^{th}$ iteration

<h2>Iterations</h2>

After initialization, 
* we estimate the next $Y_i$ from the prevous $Z_{i-1}$ (algorithm 2 in <a href="https://arxiv.org/pdf/2109.13098">paper</a>)
* and then we use that $Y_i$ to estimate the next $Z_i$ (algorithm 1 in <a href="https://arxiv.org/pdf/2109.13098">paper</a>)

The iterations continue for a fixed number of iterations (a hyperparameter), or when Y doesn't change (much) from one iteration to the next.

<h3>Algorithm 1: Update $Z_i$ from $Y_i$ and $Z_{i-1}$</h3>

* Input: $G$, $Y_{i-1}$ and $Z_{i-1}$
* Output: $Z_i$

The simplest case iterates over edges in G with:

<pre>
for u,v in zip(X0,X1):
    Z[u,Y[v]] += 1/freq(Y[v])
    Z[v,Y[u]] += 1/freq(Y[u])
</pre>

freq(lab) is the number of times lab appears in Y.

<p>
The code is slightly more complicated for graphs with weighted edges:
<p>

<pre>
for u,v,w in zip(X0,X1, X2):
    Z[u,Y[v]] += w/freq(Y[v])
    Z[v,Y[u]] += w/freq(Y[u])
</pre>


<h3>Algorithm 2: Estimate $Y_i$ from $Z_i$</h3>

* Input: $Z_{i-1}$
* Output: $Y_i$

<pre>
K = Z.shape[1]
km = kmeans(Z,K)
Y = km.labels_
</pre>

We have found the implementation of kmeans in <a href="https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization">faiss</a> to be much faster
than alternatives in sklearn (including <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html">MiniBatchKMeans</a>).

<h3>Metrics</h3>

* inertia (from kmeans): let $D$ be a vector of distances from each row in $Z$ to the closest centroid.  Return $mean(D)$.
* ARI score: (for comparing labels from $Y_{i-1}$ and $Y_i$): computed with <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html">adjusted_rand_score</a>.

We observe that ARI tends to increase with iterations.  

Early termination: stop iterating when ARI is (nearly) 1.
This is likely to happen quickly, when $Y_0$ and $Z_0$ are initialized well.
In general, ARI scores tend to improve (increase) with iterations.

Inertia tends to depend on the initialization of $Y_0$ and $Z_0$, as well as
$K$.  It is not clear why, but inertia does not seem to improve
(decrease) with iterations.  The answer may depend on a
hyperparameter: max_points_per_centroid.

<h2>Initialization</h2>

* cold start: set $Z_0$ to a matrix of zeros, and $Y_0$ to a random vector of labels: Y = np.random.choice(K, |V|)
* ProNE: set $Z_0$ to an embedding from ProNE (perhaps using a subset of G and fewer hidden dimensions).  If the ProNE embedding has fewer rows and/or columns than what is requested
for $Z_0$, fill in the extra rows and columns with zeros.

We have found that ARI scores tend to be better if we start with ProNE than if we start from a cold start.

<h3>New Cold Start</h3>

The code supports an option for a new cold start, which uses a simple heuristic to improve the chances
that vertices near one another in G will receive the same label.  Start by setting all the values in Y to -1 (unassigned).
Then iterate over the edges, assigning both vertices in the edge to the same (random) label (when possible).  That is, if they are both unassigned, then
assign them to the same random value between $0$ and $K-1$.  If one is assigned and the other is not, then fill in the missing value
with the non-miasing value.  At the end of the iteration, all the values in Y should be assigned to a value between $0$ and $K-1$.

<pre>
Y = -np.ones(|V|)	# initialize Y to -1 (unassigned)
for u,v in E:
    # Both nodes are unassigned
    if Y[u] < 0 and Y[v] < 0: Y[u] = Y[v] = np.random(K, 1)
    # One node is assigned and one is not
    if (Y[u] > 0) != (Y[v] > 0): Y[u] = Y[v] = max(Y[u], Y[v])
assert np.sum(Y < 0) == 0, 'Expected all values in Y to be assigned'
</pre>

Thus, we have three methods for initializing $Y_0$ and $Z_0$:

 1. cold start
 1. new cold start
 1. ProNE

Empirically, we obtain the best ARI scores (after the final iteration), if we start with ProNE.  The new cold start is better than the original cold start,
but not as good as ProNE, even if we computed ProNE from a smaller graph, and use fewer hidden dimensions.

<h2>Incremental Upates</h2>

Suppose we have computed $Z_{G_i}$ from a previous graph $G_i$.  Since then, we have a new graph, $G_{i+1}$, that is similar to $G_i$, though
there may be a few additional edges, and a few edges may have changed.  To obtain a quick-and-dirty estimate of $Z_{G_{i+1}}$, we recommend running GEE on $G_{i+1}$, but initialize $Z_0$ with $Z_{G_i}$.
When there is more time, we recommend running ProNE on ${G_{i+1}}$ to obtain better estimes of $Z_{G_{i+1}}$.



