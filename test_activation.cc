#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

#define EIGEN_USE_MKL_ALL
#include "Dense"

using Eigen::Matrix;
using Eigen::MatrixXf;
using std::string;
using std::cout;
using std::endl;

MatrixXf x[10];
MatrixXf y[10];
MatrixXf y_[10];

void activation(MatrixXf *neurons, const string &acttype);
void activation_opt(MatrixXf *neurons, const string &acttype);

int main (const int argv, const char **argc) {
  int ret = 0;
  int batch_size = atoi(argc[1]);

  // Eigen::initParallel();
  // mkl_set_num_threads(1);

  fprintf(stdout, "Initializing ...  0%%\n");
  x[0] = MatrixXf::Random(511, 1455);
  x[1] = MatrixXf::Random(255, 511);
  x[2] = MatrixXf::Random(127, 255);
  x[3] = MatrixXf::Random(127, 127);
  x[4] = MatrixXf::Random(127, 127);
  x[5] = MatrixXf::Random(1, 127);
  y[0] = MatrixXf::Random(1455, batch_size);
  y[1] = MatrixXf::Random(511, batch_size);
  y[2] = MatrixXf::Random(255, batch_size);
  y[3] = MatrixXf::Random(127, batch_size);
  y[4] = MatrixXf::Random(127, batch_size);
  y[5] = MatrixXf::Random(127, batch_size);
  y[6] = MatrixXf::Random(1, batch_size);
  y_[0] = MatrixXf::Random(1455, batch_size);
  y_[1] = MatrixXf::Random(511, batch_size);
  y_[2] = MatrixXf::Random(255, batch_size);
  y_[3] = MatrixXf::Random(127, batch_size);
  y_[4] = MatrixXf::Random(127, batch_size);
  y_[5] = MatrixXf::Random(127, batch_size);
  y_[6] = MatrixXf::Random(1, batch_size);

  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  string acttype = "relu";
  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 6; ++j) {
      clock_gettime(CLOCK_MONOTONIC, &ts1);
      activation(&(y[j]), acttype);
      clock_gettime(CLOCK_MONOTONIC, &ts2);
      exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
      fprintf(stdout, "Delay(%d): %.2lf us", j, exec_ns / 1e3);

      clock_gettime(CLOCK_MONOTONIC, &ts1);
      activation_opt(&(y_[j]), acttype);
      clock_gettime(CLOCK_MONOTONIC, &ts2);
      exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
      fprintf(stdout, " vs %.2lf us\n", exec_ns / 1e3);
      // fprintf(stdout, "Perf(%d): %.2lf GFlopS\n", j, 2.0 * x[j].rows() * x[j].cols() * batch_size / (exec_ns));
    }
  }

  return ret;
}

inline double Tanh(double x) {
  if (x < -15)
    return -1;
  if (x > 15)
    return 1;
  return (exp(2*x)-1.0)/(exp(2*x)+1.0);
}

inline double Sigmoid(double x) {
  if (x < -30)
    return 1e-7;
  if (x > 30)
    return 1;
  return exp(x)/(exp(x)+1.0);
}

inline double Relu(double x) {
  return x < 0 ? 0 : x;
}

void activation(MatrixXf *neurons, const string &acttype) {
  int row = neurons->rows();
  int col = neurons->cols();

  if (acttype == "relu") {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        (*neurons)(i, j) = Relu((*neurons)(i, j));
      }
    }
  } else if (acttype == "sigmoid") {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        (*neurons)(i, j) = Sigmoid((*neurons)(i, j));
      }
    }
  } else if (acttype == "tanh") {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        (*neurons)(i, j) = Tanh((*neurons)(i, j));
      }
    }
  }
  // for (int i = 0; i < row; ++i) {
  //   for (int j = 0; j < col; ++j) {
  //     if (acttype == "relu") {
  //       (*neurons)(i, j) = Relu((*neurons)(i, j));
  //     } else if (acttype == "sigmoid") {
  //       (*neurons)(i, j) = Sigmoid((*neurons)(i, j));
  //     } else if (acttype == "tanh") {
  //       (*neurons)(i, j) = Tanh((*neurons)(i, j));
  //     }
  //   }
  // }
}

void activation_opt(MatrixXf *neurons, const string &act_type) {
  if (act_type == "relu") {
    (*neurons) = neurons->cwiseMax(0);
  } else if (act_type == "sigmoid") {
    (*neurons) = 1 / (1 + (-(*neurons)).array().exp());
  } else {
    // (*neurons) = neurons->array().tanh();
    // LOG(WARNING) << "invalid activation type, act_type = " << act_type;
  }
}


