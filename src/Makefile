CXX ?= g++
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
OPENMP ?= -fopenmp
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CXXFLAGS=-fopenmp LDFLAGS=-fopenmp CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_CFLAGS ?= -O3 -fPIC $(OPENMP) -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave -DNDEBUG -DEIGEN_NO_DEBUG  -I$(BOOSTDIR)
LIBSVM_OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2/octave -I..

export

.PHONY: all clean octave libsvm liblinear

all: octave libsvm liblinear

profile: octave.profile libsvm liblinear

octave.profile:
	cd octave && make profile

octave:
	cd octave && make

libsvm:
	cd libsvm-3.17/matlab && make OCTAVE_CFLAGS="$(LIBSVM_OCTAVE_CFLAGS)" octave

liblinear:
	cd liblinear-1.94/matlab && make OCTAVE_CFLAGS="$(LIBSVM_OCTAVE_CFLAGS)" octave

clean:
	cd liblinear-1.94/matlab && make clean
	cd libsvm-3.17/matlab && make clean
	cd octave && make clean
	cd c && make clean
