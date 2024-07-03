#include <stdio.h>
#include "util.h"
#include <memory.h>
#include <math.h>
#include <errno.h>
#include <strings.h>
#include <stdlib.h>

int verbose = 0;

void usage(FILE *err, int ac, char **av)
{
  int i;
  for(i=1;i<ac;i++)
    fprintf(err, "av[%d] = %s\n", i, av[i]);
  fatal("usage: GEE (Graph Embedding Encoder): X0 X1 X2 Y Zprev Zerr Zout", err);
}

int my_max(int *vec, long n)
{
  int *end = vec + n;
  int res = vec[0];
  for(; vec<end; vec++)
    if(*vec > res) res=*vec;
  return res;
}

int *bincount(int *vec, long n, long *nres, FILE *err)
{
  int *end = vec + n;
  *nres = my_max(vec, n)+1;
  int *res = (int *)malloc(sizeof(int) * *nres);
  if(!res) fatal("malloc failed", err);
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

  if(ac != 8) usage(stderr, ac, av);

  FILE *Zout = fopen(av[7], "w");
  if(!Zout) {
    fprintf(stderr, "cannot open %s (Zout file)\n", av[7]);
    fatal("open failed", stderr);
  }

  FILE *err = fopen(av[6], "w");
  if(!err) {
    fprintf(stderr, "cannot open %s (err file)\n", av[6]);
    fatal("open failed", stderr);
  }

  if(verbose) {
    fprintf(err, "starting err file\n");
    fflush(err);
  }

  for(i=1;i<7;i++)
    if(!file_exists(av[i])) {
      fprintf(err, "file not found: %s\n", av[i]);
      fatal("assertion failed", err);
    }

  int *X0 = (int *)mmapfile(av[1], &nX0, err);
  nX0 /= sizeof(int);
  fprintf(err, "X0: len(%s) = %ld ints (s)\n", av[1], nX0);
  fflush(err);

  int *X1 = (int *)mmapfile(av[2], &nX1, err);
  nX1 /= sizeof(int);
  fprintf(err, "X1: len(%s) = %ld ints (s)\n", av[2], nX1);
  fflush(err);

  float *X2 = (float *)mmapfile(av[3], &nX2, err);
  nX2 /= sizeof(float);
  fprintf(err, "X2: len(%s) = %ld floats (s)\n", av[3], nX2);
  fflush(err);

  if(nX0 != nX1 || nX0 != nX2) fatal("assertion failed", err);
  s = nX0;

  int *Y = (int *)mmapfile(av[4], &n, err);
  n /= sizeof(int);  
  fprintf(err, "Y: len(%s) = %ld ints (n)\n", av[4], n);
  fflush(err);

  int *X0end = X0 + s;
  int *nk = bincount(Y, n, &k, err);

  print_bincount(err, nk, k);

  double Ymean = 1.0;
  // double Ymean = n/(double)k;

  fprintf(err, "s (edges) = %ld, n = %ld, k = %ld\n", s, n, k);
  fflush(err);

  fprintf(err, "about to read Z from %s\n", av[5]);
  fflush(err);

  long nZ;
  float *Z = (float *)malloc(sizeof(float) * n * k);
  if(!Z) fatal("malloc failed", err);

  if(!file_exists(av[5])) {
    nZ = n * k;
    fprintf(err, "initializing Z to %ld zeros\n", nZ);
    fflush(err);
    memset(Z, 0, sizeof(float) * nZ);
  }
  else {    
    float *Z = (float *)readchars(av[5], &nZ, err);
    nZ /= sizeof(float);
    fprintf(err, "Z: %ld floats in %s (Z)\n", nZ, av[5]);
    fflush(err);
  }

  if(nZ != n * k) {
    fprintf(err, "nZ = %ld, n = %ld, k = %ld\n", nZ, n, k);
    fflush(err);
    fatal("assertion failed", err);
  }
    
  // float *Z = (float *)malloc(sizeof(float) * n * k);
  // if(!Z) fatal("malloc failed");

  // memset(Z, 0, sizeof(float) * n * k);
  // fprintf(stderr, "pt1\n");

  long ntick = 100;
  long tick = (X0end - X0)/ntick;
  int *X0tick = X0 + tick;
  int *X0base = X0;

  fprintf(err, "tick = %ld, ntick = %ld\n", tick, ntick);
  fflush(err);

  for(;X0 < X0end; X0++,X1++,X2++) {

    if(verbose) {
      long gap = X0 - X0base;
      fprintf(err, "gap = %ld\n", gap);
      fflush(err);
    }

    if(X0 >= X0tick) {
      fprintf(err, "%f done\n", (X0 - X0base)/(double)ntick);
      fflush(err);
      X0tick += tick;
    }

    if(verbose) {
      fprintf(err, "pt 1\n");
      fflush(err);
    }

    int v_i = *X0;

    if(verbose) {
      fprintf(err, "v_i = %d\n", v_i);
      fflush(err);
    }

    int v_j = *X1;

    if(verbose) {
      fprintf(err, "v_j = %d\n", v_j);
      fflush(err);
    }

    if(v_i < 0 || v_i >= n) {
      fprintf(err, "v_i = %d, assertion failed\n", v_i);
      fflush(err);
      fatal("assertion failed", err);
    }

    if(v_j < 0 || v_j >= n) {
      fprintf(err, "v_j = %d, assertion failed\n", v_j);
      fflush(err);
      fatal("assertion failed", err);
    }

    if(verbose) {
      fprintf(err, "pt 2\n");
      fflush(err);
    }

    int label_i = Y[v_i];

    if(verbose) {
      fprintf(err, "label_i = %d\n", label_i);
      fflush(err);
    }

    int label_j = Y[v_j];

    if(verbose) {
      fprintf(err, "label_j = %d\n", label_j);
      fflush(err);
    }

    if(label_i < 0 || label_i >= k) {
      fprintf(err, "label_i = %d, assertion failed\n", label_i);
      fflush(err);
      fatal("assertion failed", err);
    }

    if(label_j < 0 || label_j >= k) {
      fprintf(err, "label_j = %d, assertion failed\n", label_j);      
      fflush(err);
      fatal("assertion failed", err);
    }
    
    // fprintf(stderr, "edge: %d, %d; labels = %d, %d\n", v_i, v_j, label_i, label_j);

    long o = v_i *k + label_j ;

    if(verbose) {
      fprintf(err, "o = %ld, nZ = %ld\n", o, nZ);
      fflush(err);
    }

    if(o < 0 || o >= nZ) {
      fprintf(err, "o is out of range: o = %ld, nZ = %ld\n", o, nZ);
      fflush(err);
      fatal("assertion failed", err);
    }

    if(verbose) {
      fprintf(err, "pt 3\n");
      fflush(err);

    fprintf(err, "nk[%d] = %d\n", label_j, nk[label_j]);
    fflush(err);

    fprintf(err, "*X2 = %f\n", *X2);
    fflush(err);

    fprintf(err, "Z[o] = %f\n", Z[o]);
    fflush(err);
    }
    
    Z[o] += (Ymean * *X2)/(double)nk[label_j]; /* Ymean added by kwc */

    if(verbose) {
      fprintf(err, "pt 4\n");
      fflush(err);
    }

    if(v_i != v_j) {
      long o2 = v_j*k + label_i;
      if(o2 < 0 || o2 >= nZ) {
	fprintf(err, "o2 is out of range: o2 = %ld, nZ = %ld\n", o2, nZ);
	fflush(err);
	fatal("assertion failed", err);
      }
      Z[o2] += *X2/(double)nk[label_i];
    }

    if(verbose) {
      fprintf(err, "pt 5\n");
      fflush(err);
    }
  }

  // normalize(Z, n, k);

  fprintf(err, "about to output\n");
  fflush(err);

  if(fwrite(Z, sizeof(float), n*k, Zout) != n*k)
    fatal("write failed", err);

  fprintf(err, "GEE done\n");
  fflush(err);

  return 0;
}
