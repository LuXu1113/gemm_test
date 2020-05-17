#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

#define MAX_MATRIX_NUM (2)
#define MAX_MATRIX_SIZE (2048)
#define TEST_ROUND_NUM (100)

//  m_z = m_z * m_y
static float *m_x[MAX_MATRIX_NUM];
static float *m_y[MAX_MATRIX_NUM];
static float *m_z[MAX_MATRIX_NUM];

//  Z = X(row: m, col: k) x Y(row: k, col: n)
void test(int m, int n, int k) {
  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  CBLAS_LAYOUT    layout = CblasRowMajor;
  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasNoTrans;
  float alpha = 1.0;
  float beta  = 0.0;
  int lda = k;
  int ldb = n;
  int ldc = n;

  clock_gettime(CLOCK_MONOTONIC, &ts1);
  for (int i = 0; i < TEST_ROUND_NUM; ++i) {
    for (int j = 0; j < MAX_MATRIX_NUM; ++j) {
      float *a = m_x[j];
      float *b = m_y[j];
      float *c = m_z[j];
      cblas_sgemm(layout, transA, transB, m, n, k,
          alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts2);

  exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);

  fprintf(stdout, "Delay(avg): %.2lf us\n", exec_ns / 1e3 / TEST_ROUND_NUM / MAX_MATRIX_NUM);
  fprintf(stdout, "Perf(avg): %.2lf GFlopS\n", 2.0 * m * n * k / (exec_ns / TEST_ROUND_NUM/ MAX_MATRIX_NUM));
}

int main(const int argv, const char **argc) {
  int ret = 0;
  int m = atoi(argc[1]);
  int n = atoi(argc[2]);
  int k = atoi(argc[3]);

  // mkl_set_num_threads(1);
  fprintf(stdout, "parameters: [m: %d, n: %d, k: %d]\n", m, n, k);

  fprintf(stdout, "Initializing ...  0%%\n");
  for (int i = 0; i < MAX_MATRIX_NUM; ++i) {
    m_x[i] = (float *)mkl_calloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE, sizeof(float), 64);
    m_y[i] = (float *)mkl_calloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE, sizeof(float), 64);
    m_z[i] = (float *)mkl_calloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE, sizeof(float), 64);

    for (int j = 0; j < MAX_MATRIX_SIZE * MAX_MATRIX_SIZE; ++j) {
      m_x[i][j] = ((float)rand() / (float)(RAND_MAX) - 0.5) * 500.0;
      m_y[i][j] = ((float)rand() / (float)(RAND_MAX) - 0.5) * 500.0;
      //  m_z[i][j] = ((float)rand() / (float)(RAND_MAX) - 0.5) * 500.0;
    }

    if (i % 128 == 0) {
      fprintf(stdout, "             ... %2d%%\n", (int)((float)(i + 1) / (float)MAX_MATRIX_NUM * 100));
    }
  }

  test(m, n, k);

  return ret;
}


