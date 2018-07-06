CXX ?= g++
CC ?= gcc

THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
MAKEFILE_DIR = $(abspath $(dir $(THIS_MAKEFILE)))


BOOSTDIR ?= $(HOME)/local
BOOSTINC ?= -I$(BOOSTDIR)/include
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK ?= -L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 


CROARINGDIR ?= $(MAKEFILE_DIR)/../CRoaring
CROARINGINC ?= -I$(CROARINGDIR)
CROARINGLIB ?= $(CROARINGDIR)
CROARINGLINK:=-L$(CROARINGLIB) -Wl,-rpath,$(CROARINGLIB) 

BINDIR ?= bin

# OMP options
# 1. full parallelization 
OPENMP ?= -fopenmp -DMCTHREADS=1
# for valgrind debug, compile without openmp... then can see "0 errors from 0 contexts"
# 2. no openmp anywhere
#OPENMP:= -Wno-error=unknown-pragmas -Wno-error=unused-variable
# 3. allow Eigen/blas threads, but no MCFilter source code threads
#OPENMP:= -fopenmp -DMCTHREADS=0
# 4. allow MCFilter threads but not Eigen threads
#OPENMP:= -fopenmp -DEIGEN_DONT_PARALLELIZE
# allow STL experimental threads?


CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG $(BOOSTINC) $(CROARINGINC) -std=c++11 -pipe -ggdb3 -fPIC

LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring


PROFILE_OPENMP = 
PROFILE_OPENMP += -fopenmp
PROFILE_OPENMP += -DMCTHREADS=0
#PROFILE_OPENMP += -DEIGEN_DONT_PARALLELIZE

PROFILEDIR = $(HOME)/local
PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP) -std=c++11 -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include $(BOOSTINC) $(CROARINGINC) 
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 


NCPUS:=$(shell cat /proc/cpuinfo | grep '^processor' | wc -l)
JOBS:=-j$(NCPUS)

export

C_TARGETS = mcproj mcsolve clean lib

PROFILE_TARGETS = $(patsubst %,profile_%,$(C_TARGETS))

.PHONY: all clean lib CRoaring clean-all CRoaring-bld

all: CRoaring $(C_TARGETS)

$(C_TARGETS):
	mkdir -p $(BINDIR)
	$(MAKE) -C $(BINDIR) $(JOBS) -f../c/Makefile $@

#enable profiling
$(PROFILE_TARGETS):
	mkdir -p $(BINDIR)-profile
	$(MAKE) -C $(BINDIR)-profile CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" -f../c/Makefile $(patsubst profile_%,%,$@) 


CRoaring:
	$(MAKE) -C $(CROARINGDIR) -f ../src/Makefile CRoaring-bld

CRoaring-bld:
	./amalgamation.sh 
	$(CC) -march=native -O3 -std=c11 -o roaring.o -c roaring.c
	$(AR) -vru libroaring.a roaring.o
	$(CC) -O3 -std=c11 -shared -o libroaring.so -fPIC -fPIC roaring.c

CRoaring-clean:
	cd $(CROARINGDIR) && rm roaring.o libroaring.a libroaring.so roaring.c roaring.h roaring.hh

clean-all: clean CRoaring-clean
