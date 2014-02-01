CXX ?= g++

OCTAVEDIR ?= $(HOME)/Programs/octave-3.6.4
OCTAVE_MEX ?= env CC=$(CXX)  $(OCTAVEDIR)/bin/mkoctfile -v -g -I/home/mlshack/alex/Programs/gperftools-2.1/install/include -L/home/mlshack/alex/Programs/gperftools-2.1/install/lib -lprofiler
OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.6.4/octave -I..

export

.PHONY: all clean octave libsvm

all: octave libsvm

octave:
	cd octave && make

libsvm:
	cd libsvm-3.17/matlab && make octave

clean:
	cd libsvm-3.17/matlab && make clean
	cd octave && make clean
