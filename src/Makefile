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

#using gperftools tcmalloc and profiler. Change this to point to your instalation of gperftools 
GPERFTOOLSDIR ?= $(HOME)/local
GPERFTOOLSINC ?= -I$(GPERFTOOLSDIR)/include
GPERFTOOLSLIB ?=  $(GPERFTOOLSDIR)/lib
GPERFTOOLSLINK ?= -L$(GPERFTOOLSLIB) -Wl,-rpath,$(GPERFTOOLSLIB)

BINDIR ?= $(MAKEFILE_DIR)/../bin

# OMP options
# 1. full parallelization 
OPENMP ?= -fopenmp -DMCTHREADS=1
# for valgrind debug, compile without openmp... then can see "0 errors from 0 contexts"
# 2. no openmp anywhere
# OPENMP:= -Wno-error=unknown-pragmas -Wno-error=unused-variable
# 3. allow Eigen/blas threads, but no MCFilter source code threads
#OPENMP:= -fopenmp -DMCTHREADS=0
# 4. allow MCFilter threads but not Eigen threads
#OPENMP:= -fopenmp -DEIGEN_DONT_PARALLELIZE
# allow STL experimental threads?

CXX ?= g++
CC ?= gcc


CPPFLAGS ?= -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG $(BOOSTINC) $(CROARINGINC) $(EIGENINC) -march=native -ffast-math -m64  -std=c++11 -pipe -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free 
LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring $(GPERFTOOLSLINK) -ltcmalloc 


#profileing settings 
PROFILE_OPENMP = 
PROFILE_OPENMP += -fopenmp
PROFILE_OPENMP += -DMCTHREADS=1
#PROFILE_OPENMP += -DEIGEN_DONT_PARALLELIZE

PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP)  -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE $(GPERFTOOLSINC) $(BOOSTINC) $(CROARINGINC) $(EIGENINC) -march=native -ffast-math -m64 -std=c++11 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
PROFILE_LDFLAGS = $(LDFLAGS) -lprofiler 


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

all: lib mcproj mcsolve 

$(C_TARGETS):
	mkdir -p $(BINDIR)
	$(MAKE) -C $(BINDIR) $(JOBS) -f../src/c/Makefile $@

#enable profiling
$(PROFILE_TARGETS):
	mkdir -p $(BINDIR)-prof
	$(MAKE) -C $(BINDIR)-prof CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" -f../src/c/Makefile $(patsubst %-prof,%,$@) 

#enable debug
$(DBG_TARGETS):
	mkdir -p $(BINDIR)-dbg
	$(MAKE) -C $(BINDIR)-dbg CPPFLAGS="$(DBGFLAGS)" -f../c/Makefile $(patsubst %-dbg,%,$@) 


CRoaring:
	$(MAKE) -C $(CROARINGDIR) -f ../src/Makefile CRoaring-bld

CRoaring-bld:
	./amalgamation.sh 
	$(CC) -O3-march=native -std=c11 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -o roaring.o -c roaring.c 
	$(AR) -Uvru libroaring.a roaring.o
	$(CC) -O3 -march=native -std=c11 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -shared -o libroaring.so -fPIC roaring.c

CRoaring-prof:
	$(MAKE) -C $(CROARINGDIR) -f ../src/Makefile CRoaring-bld-prof

CRoaring-bld-prof:
	./amalgamation.sh 
	$(CC) -ggdb3 -O3 -march=native -std=c11 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -o roaring.o -c roaring.c
	$(AR) -Uvru libroaring.a roaring.o
	$(CC) -ggdb3 -O3 -march=native -std=c11 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -shared -o libroaring.so -fPIC roaring.c

CRoaring-clean:
	cd $(CROARINGDIR) && rm roaring.o libroaring.a libroaring.so roaring.c roaring.h roaring.hh

clean-all: clean CRoaring-clean

