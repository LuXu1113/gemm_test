#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define EIGEN_USE_MKL_ALL
#include "Dense"

#define MAX_MATRIX_NUM (100)
#define MAX_MATRIX_SIZE (2048)
#define TEST_ROUND_NUM (100)

using Eigen::Matrix;
using Eigen::MatrixXf;
using std::cout;
using std::endl;

MatrixXf x[MAX_MATRIX_NUM];
MatrixXf y[MAX_MATRIX_NUM];
MatrixXf z[MAX_MATRIX_NUM];

int main (const int argv, const char **argc) {
  int ret = 0;
  int m = atoi(argc[1]);
  int n = atoi(argc[2]);
  int k = atoi(argc[3]);

  mkl_set_num_threads(1);
  fprintf(stdout, "parameters: [m: %d, n: %d, k: %d]\n", m, n, k);

  fprintf(stdout, "Initializing ...  0%%\n");
  for (int i = 0; i < MAX_MATRIX_NUM; ++i) {
    x[i] = MatrixXf::Random(m, k);
    y[i] = MatrixXf::Random(k, n);
    z[i] = MatrixXf::Random(m, n);
    if (i % 128 == 0) {
      fprintf(stdout, "             ... %2d%%\n", (int)((float)(i + 1) / (float)MAX_MATRIX_NUM * 100));
    }
  }

  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  for (int i = 0; i < TEST_ROUND_NUM; ++i) {
    for (int j = 0; j < MAX_MATRIX_NUM; ++j) {
      z[j].noalias() = x[j] * y[j];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts2);

  exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);

  fprintf(stdout, "Delay(avg): %.2lf us\n", exec_ns / 1e3 / TEST_ROUND_NUM / MAX_MATRIX_NUM);
  fprintf(stdout, "Perf(avg): %.2lf GFlopS\n", 2.0 * m * n * k / (exec_ns / TEST_ROUND_NUM/ MAX_MATRIX_NUM));

  return ret;
}

