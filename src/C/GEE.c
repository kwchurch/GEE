#include <stdio.h>
#include "util.h"
#include <memory.h>
#include <math.h>
#include <errno.h>
#include <strings.h>
#include <stdlib.h>

void usage()
{
  fatal("usage: GEE (Graph Embedding Encoder): X0 X1 X2 Y Zprev Zerr > Z");
}

int my_max(int *vec, long n)
{
  int *end = vec + n;
  int res = vec[0];
  for(; vec<end; vec++)
    if(*vec > res) res=*vec;
  return res;
}

int *bincount(int *vec, long n, long *nres)
{
  int *end = vec + n;
  *nres = my_max(vec, n)+1;
  int *res = (int *)malloc(sizeof(int) * *nres);
  if(!res) fatal("malloc failed");
  memset(res, 0, sizeof(int) * *nres);
  for(; vec<end;vec++)
    res[*vec]++;

  return res;
}

void normalize_row(float *row, long k)
{
  long i;
  double d = 0;
  for(i=0;i<k;i++) d += row[i]*row[i];
  for(i=0;i<k;i++) row[i] /= d;
}

void normalize(float *Z, long n, long k)
{
  float *Zend = Z + n * k;
  for( ; Z < Zend; Z += k)
    normalize_row(Z, k);
}

void print_bincount(FILE *fd, int *nk, long k)
{
  int i;
  for(i=0;i<k;i++)
    fprintf(fd, "bin[%d] = %d\n", i, nk[i]);
  fflush(fd);
}

int main(int ac, char **av)
{
  long nX0, nX1, nX2;
  long s;			/* number of edges (X values) */
  long k;			/* number of hidden dimensions */
  long n;			/* number of labels (Y values) */
  int i;

  if(ac != 7) usage();

  for(i=1;i<6;i++)
    if(!file_exists(av[i])) {
      fprintf(stderr, "file not found: %s\n", av[i]);
      fatal("assertion failed");
    }
      
  FILE *err = fopen(av[6], "w");

  
  int *X0 = (int *)mmapfile(av[1], &nX0);
  nX0 /= sizeof(int);
  int *X1 = (int *)mmapfile(av[2], &nX1);
  nX1 /= sizeof(int);

  float *X2 = (float *)mmapfile(av[3], &nX2);
  nX2 /= sizeof(float);

  if(nX0 != nX1 || nX0 != nX2) fatal("assertion failed");
  s = nX0;

  int *Y = (int *)mmapfile(av[4], &n);
  n /= sizeof(int);  

  int *X0end = X0 + s;
  int *nk = bincount(Y, n, &k);

  print_bincount(err, nk, k);

  /* float *nk2 = (float *)malloc(sizeof(float) * n); */
  /* if(!nk2) fatal("malloc failed"); */
  /* int i; */
  /* for(i=0;i<n;i++) nk2[i] = nk[Y[i]]; */

  fprintf(err, "s (edges) = %ld, n = %ld, k = %ld\n", s, n, k);
  fflush(err);

  fprintf(err, "about to read Z from %s\n", av[5]);
  fflush(err);

  long nZ;
  float *Z = (float *)readchars(av[5], &nZ);
  nZ /= sizeof(float);

  fprintf(err, "found %ld floats\n", nZ);
  fflush(err);

  if(nZ != n * k) {
    fprintf(err, "nZ = %ld, n = %ld, k = %ld\n", nZ, n, k);
    fflush(err);
    fatal("assertion failed");
  }
    
  // float *Z = (float *)malloc(sizeof(float) * n * k);
  // if(!Z) fatal("malloc failed");

  // memset(Z, 0, sizeof(float) * n * k);
  // fprintf(stderr, "pt1\n");

  long ntick = 1000000;
  long tick = (X0end - X0)/ntick;
  int *X0tick = X0 + tick;
  int *X0base = X0;

  fprintf(err, "tick = %ld, ntick = %ld\n", tick, ntick);
  fflush(err);

  for(;X0 < X0end; X0++,X1++,X2++) {

    // long gap = X0 - X0base;
    // fprintf(err, "gap = %ld\n", gap);
    // fflush(err);

    if(X0 >= X0tick) {
      fprintf(err, "%f done\n", (X0 - X0base)/(double)ntick);
      fflush(err);
      X0tick += tick;
    }

    int v_i = *X0;
    int v_j = *X1;

    if(v_i < 0 || v_i >= n) {
      fprintf(err, "v_i = %d, assertion failed\n", v_i);
      fflush(err);
      fatal("assertion failed");
    }

    if(v_j < 0 || v_j >= n) {
      fprintf(err, "v_j = %d, assertion failed\n", v_j);
      fflush(err);
      fatal("assertion failed");
    }

    int label_i = Y[v_i];
    int label_j = Y[v_j];

    if(label_i < 0 || label_i >= k) {
      fprintf(err, "label_i = %d, assertion failed\n", label_i);
      fflush(err);
      fatal("assertion failed");
    }

    if(label_j < 0 || label_j >= k) {
      fprintf(err, "label_j = %d, assertion failed\n", label_j);      
      fflush(err);
      fatal("assertion failed");
    }
    
    // fprintf(stderr, "edge: %d, %d; labels = %d, %d\n", v_i, v_j, label_i, label_j);

    long o = v_i *k + label_j ;
    if(o < 0 || o >= nZ) {
      fprintf(err, "o is out of range: o = %ld, nZ = %ld\n", o, nZ);
      fflush(err);
      fatal("assertion failed");
    }

    Z[o] += *X2/(double)nk[label_j];
    if(v_i != v_j) {
      long o2 = v_j*k + label_i;
      if(o2 < 0 || o2 >= nZ) {
	fprintf(err, "o2 is out of range: o2 = %ld, nZ = %ld\n", o2, nZ);
	fflush(err);
	fatal("assertion failed");
      }
      Z[o2] += *X2/(double)nk[label_i];
    }
  }

  // normalize(Z, n, k);

  fprintf(err, "about to output\n");
  fflush(err);

  if(fwrite(Z, sizeof(float), n*k, stdout) != n*k)
    fatal("write failed");

  return 0;
}
