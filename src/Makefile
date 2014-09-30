CXX ?= g++
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v
OCTAVE_CFLAGS ?= -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
LIBSVM_OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2/octave -I..

export

.PHONY: all clean octave libsvm

all: octave libsvm

octave:
	cd octave && make

libsvm:
	cd libsvm-3.17/matlab && make OCTAVE_CFLAGS="$(LIBSVM_OCTAVE_CFLAGS)" octave

clean:
	cd libsvm-3.17/matlab && make clean
	cd octave && make clean
	cd c && make clean
