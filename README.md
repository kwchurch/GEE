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

Let G=(V,E) be a graph
and Z be an embedding with K hidden dimensions.
In other words, Z is an array with dtype of np.float32 and shape: (|V|, K)

Cosines of rows of Z can be interpreted in terms of random walks on G.

Y is a sequence of |V| class labels, where 0 <= Y[i] < K.  
Note: K is both the number
of hidden dimensions in Z as well as the number of class lables in Y.

G will be represented with X0, X1, X2.  All three vectors have length of |E|.
Edges go from x0 in X0 to x1 in X2 with weights x3 in X3.
X0 and X1 are stored with dtype of np.int32, and X2 is stored with dtype of np.float32
X2 is optional, and defaults to a vector of ones (if not specified).

Inputs to GEE:
* G (required)
* options for initializing Z (either from cold start or from something better)
* options for restarting iterations
* optional hyper parameters

Outputs:
* Z: embedding
* Y: class labels

<h2>Iterations</h2>

After initialization, 
* we estimate the next Y from the prevous Z (algorithm 2 in <a href="https://arxiv.org/pdf/2109.13098">paper</a>)
* and then we use that Y to estimate the next Z (algorithm 1 in <a href="https://arxiv.org/pdf/2109.13098">paper</a>)

The iterations continue for a fixed number of iterations (a hyperparameter), or when Y doesn't change (much) from one iteration to the next.

<h3>Algorithm 1: Update Z from Y</h3>

* Input: G, Y and Z from a previous iteration
* Output: an updated estimate for Z 

The simplest case iterates over edges in G with:

<pre>
for u,v in E:
    Z[u,Y[v]] += 1/freq(Y[v])
    Z[v,Y[u]] += 1/freq(Y[u])
</pre>

freq(lab) is the number of times lab appears in Y.

<p>
The code is slightly more complicated for graphs with weighted edges.
<p>


<h3>Algorithm 2: Estimate Y from Z</h3>

* Input: Z
* Output: Y

<pre>
K = Z.shape[1]
km = kmeans(Z,K)
Y = km.labels_
</pre>

We have found the implementation of kmeans in <a href="https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization">faiss</a> to be much faster
than alternatives in sklearn (including <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html">MiniBatchKMeans</a>).

<h3>Metrics</h3>

* RMS error (from kmeans): let D be a vector of distances from each row in Z to the closest centroid.  Return sqrt(mean($D^2$)).
* ARI score: (for comparing labels from Y to labels from the previous estimate of Y): computed with <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html">adjusted_rand_score</a>.

We observe that ARI tends to increase with iterations.  

Early termination: stop iterating when ARI is (nearly) 1.
This is likely to happen quickly, when Y and Z are initialized well.
In general, ARI scores tend to improve (increase) with iterations.

RMS error tends to depend on the initialization of Y and Z, as well as K.
RMS error does not seem to improve (decrease) with iterations.

<h2>Initialization</h2>

* cold start: set Z to a matrix of zeros, and Y to a random vector of labels: Y = np.random.choice(K, |V|)
* ProNE: set Z to an embedding from ProNE (perhaps using a subset of G and fewer hidden dimensions).  If the ProNE embedding has fewer rows and/or columns than what is requested
for the output Z, fill in the extra rows and columns with zeros.

We have found that ARI scores tend to be better if we start with ProNE than if we start from a cold start.
The code supports an option for a new cold start, which uses a simple heuristic to improve the chances
that vertices near one another in G will receive the same label.

<pre>
Y = -np.ones(|V|)
for u,v in E:
    if Y[u] < 0 and Y[v] < 0: Y[u] = Y[v] = np.random(K, 1)
    if (Y[u] > 0) != (Y[v] > 0): Y[u] = Y[v] = max(Y[u], Y[v])
</pre>

Thus, we have three methods for initializing Y and Z:

 1. cold start
 1. new cold start
 1. ProNE

Empirically, we obtain the best ARI scores (after the final iteration), if we start with ProNE.  The new cold start is better than the original cold start,
but not as good as ProNE, even if we computed ProNE from a smaller graph, and use fewer hidden dimensions.

<h2>Incremental Upates</h2>

Suppose we have computed $Z_i$ from a previous graph $G_i$.  Since then, we have a new graph, $G_{i+1}$, that is similar to $G_i$, though
there may be some additional edges, and some edges may have changed.  We recommend running GEE on $G_{i+1}$, but initialize with $Z_i$.

