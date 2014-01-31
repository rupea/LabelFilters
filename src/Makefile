CXX ?= g++

OCTAVEDIR ?= ~/Programs/octave-3.6.4/
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile
OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.6.4/octave/ -I..

.PHONY: all clean octave libsvm

all: octave libsvm

octave:
	cd octave; make

libsvm:
	cd libsvm-3.17/matlab; make octave

clean:
	cd libsvm-3.17/matlab; make clean
	cd octave; make clean
