all: test_mkl test_eigen test_activation
.PHONY: all

CC = gcc
CPP = g++

OPTIMIZE = -O3 -DUSE_TCMALLOC=1 -MMD -msse3 

MKL_LIBRARY = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_sequential -lmkl_core -lstdc++ -lpthread -lm -lrt -ldl -lgomp
EIGEN_LIBRARY = -lrt

MKL_FLAGS = $(MKL_LIBRARY) $(OPTIMIZE)
EIGEN_FLAGS = $(MKL_LIBRARY) $(OPTIMIZE)

test_mkl: test_mkl.cc
	$(CPP) test_mkl.cc $(MKL_FLAGS) -o test_mkl

test_eigen: test_eigen.cc
	$(CPP) test_eigen.cc $(EIGEN_FLAGS) -o test_eigen

test_activation: test_activation.cc
	$(CPP) test_activation.cc  $(EIGEN_FLAGS) -lrt -o test_activation

.PHONY: clean

clean:
	rm -f test_mkl test_eigen test_activation *.d

