CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
OPENMP ?= -fopenmp
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CXXFLAGS=-fopenmp LDFLAGS=-fopenmp CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_CFLAGS ?= -O3 -fPIC $(OPENMP) -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave -DNDEBUG -DEIGEN_NO_DEBUG  -I$(BOOSTDIR)
LIBSVM_OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2/octave -I..
NCPUS:=$(shell cat /proc/cpuinfo | grep '^processor' | wc -l)
JOBS:=-j$(NCPUS)

export

.PHONY: all clean octave libsvm liblinear lua libmcfilter

all: octave libsvm liblinear

# lua and libmcfilter targets do not involve octave (just c++ and milde_core's lua)
lua: libmcfilter | libmcfilter
	$(MAKE) -C lua-api $(JOBS)
libmcfilter:
	$(MAKE) -C c $(JOBS) lib

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
	$(MAKE) -C lua-api clean
