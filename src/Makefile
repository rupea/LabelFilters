THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
MAKEFILE_DIR := $(abspath $(dir $(THIS_MAKEFILE)))

# CRoaring is included as a git submodule. Change this to use your own CRoaring install
CROARINGDIR ?= $(MAKEFILE_DIR)/../CRoaring
CROARINGINC ?= -I$(CROARINGDIR)
CROARINGLIB ?= $(CROARINGDIR)
CROARINGLINK ?= -L$(CROARINGLIB) -Wl,-rpath,$(CROARINGLIB) 

# Eigen is included as a git submodule. Change this ig you want to use your own Eigen install
EIGENDIR ?= $(MAKEFILE_DIR)/../Eigen
EIGENINC ?= -I$(EIGENDIR)/

# using boost program options package. Change this to your instalation of boost
BOOSTDIR ?= $(HOME)/local
BOOSTINC ?= -I$(BOOSTDIR)/include
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK ?= -L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 


#using gperftools to profile the code. Change this to point to your instalation of gperftools 
PROFILEDIR ?= $(HOME)/local
PROFILEINC ?= -I$(PROFILEDIR)/include
PROFILELIB ?= $(PROFILEDIR)/lib
PROFILELINK ?= -L$(PROFILELIB) -Wl,-rpath,$(PROFILELIB)


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

CXX ?= g++
CC ?= gcc


CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG $(BOOSTINC) $(CROARINGINC) $(EIGENINC) -std=c++11 -pipe 
LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring


#profileing settings 
PROFILE_OPENMP = 
PROFILE_OPENMP += -fopenmp
PROFILE_OPENMP += -DMCTHREADS=0
#PROFILE_OPENMP += -DEIGEN_DONT_PARALLELIZE

PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP) -std=c++11 -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE $(PROFILEINC) $(BOOSTINC) $(CROARINGINC) $(EIGENINC)
PROFILE_LDFLAGS = $(LDFLAGS) $(PROFILELINK) -lprofiler 


DBGFLAGS ?= $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS))
DBGFLAGS += -ggdb3 -Wall -Wno-sign-compare # -Werror
# for better tracing...
# DBGFLAGS += -O1


# testing...
#MKLROOT:=/opt/intel/compilers_and_libraries/linux/mkl/
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include 
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm



NCPUS:=$(shell cat /proc/cpuinfo | grep '^processor' | wc -l)
JOBS:=-j$(NCPUS)

export

C_TARGETS = mcproj mcsolve clean lib

PROFILE_TARGETS = $(patsubst %,%-prof,$(C_TARGETS))
DBG_TARGETS = $(patsubst %,%-dbg,$(C_TARGETS))

.PHONY: all clean lib CRoaring clean-all CRoaring-bld

all: CRoaring lib mcproj mcsolve 

$(C_TARGETS):
	mkdir -p $(BINDIR)
	$(MAKE) -C $(BINDIR) $(JOBS) -f../c/Makefile $@

#enable profiling
$(PROFILE_TARGETS):
	mkdir -p $(BINDIR)-prof
	$(MAKE) -C $(BINDIR)-prof CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" -f../c/Makefile $(patsubst %-prof,%,$@) 

#enable debug
$(DBG_TARGETS):
	mkdir -p $(BINDIR)-dbg
	$(MAKE) -C $(BINDIR)-dbg CPPFLAGS="$(DBGFLAGS)" -f../c/Makefile $(patsubst %-dbg,%,$@) 


CRoaring:
	$(MAKE) -C $(CROARINGDIR) -f ../src/Makefile CRoaring-bld

CRoaring-bld:
	./amalgamation.sh 
	$(CC) -march=native -O3 -std=c11 -o roaring.o -c roaring.c
	$(AR) -Uvru libroaring.a roaring.o
	$(CC) -O3 -std=c11 -shared -o libroaring.so -fPIC roaring.c

CRoaring-clean:
	cd $(CROARINGDIR) && rm roaring.o libroaring.a libroaring.so roaring.c roaring.h roaring.hh

clean-all: clean CRoaring-clean

