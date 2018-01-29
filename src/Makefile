CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK:=-L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 

OPENMP ?= -fopenmp

ifneq ($(wildcard /usr/include/octave-3.8.2),)
OCTAVEDIR ?= /usr
BINDIR ?= bin-newmlc
OCT_INC := -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
OCT_LINK:= -Wl,-rpath,$(OCTAVEDIR)/lib64 -Wl,-rpath,$(OCTAVEDIR)/lib64/octave/3.8.2
endif
ifneq ($(wildcard $(HOME)/Programs/octave-3.8.2),)
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
BINDIR ?= bin-mlc
OCT_INC := -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
OCT_LINK:= -Wl,-rpath,$(OCTAVEDIR)/lib64 -Wl,-rpath,$(OCTAVEDIR)/lib64/octave/3.8.2
endif
ifneq ($(wildcard /usr/include/octave-3.8.1),)
OCTAVEDIR = /usr
BINDIR ?= bin_oct3.8.1
OCT_INC:=-I$(OCTAVEDIR)/include/octave-3.8.1 -I$(OCTAVEDIR)/include/octave-3.8.1/octave
OCT_LINK:= -Wl,-rpath,$(OCTAVEDIR)/lib64 -Wl,-rpath,$(OCTAVEDIR)/lib64/octave/3.8.1
endif


HDF5_INC ?= -I/usr/include/hdf5/serial
OCTAVE_CFLAGS ?= -O3 -fPIC $(OPENMP) -std=c++11 -march=native -ffast-math -m64 -pipe $(OCT_INC) $(HDF5_INC) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR)
OCTAVE_LDFLAGS ?= -L../c
OCTAVE_MEX ?= env CXXFLAGS="$(OCTAVE_CFLAGS)" LDFLAGS="$(OCTAVE_LDFLAGS)" CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
LIBSVM_OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -DNDEBUG $(OCT_INC) -I..



OPENMP ?= -fopenmp
# for valgrind debug, compile without openmp... then can see "0 errors from 0 contexts"
# 1. no openmp anywhere
#OPENMP:= -Wno-error=unknown-pragmas -Wno-error=unused-variable
# 2. allow Eigen/blas threads, but no MCFilter source code threads
#OPENMP:= -DMCTHREADS=0
# allow STL experimental threads?

CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) -std=c++11 -pipe -ggdb3 -fPIC $(OCT_INC) $(HDF5_INC)

LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(OCT_LINK)


PROFILE_OPENMP = $(OPENMP)
PROFILEDIR = $(HOME)/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP) -std=c++11 -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR) $(OCT_INC) $(HDF5_INC)
PROFILE_CPPFLAGS+=-DMCTHREADS=1
#PROFILE_CPPFLAGS += -DEIGEN_DONT_PARALLELIZE
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 


NCPUS:=$(shell cat /proc/cpuinfo | grep '^processor' | wc -l)
JOBS:=-j$(NCPUS)

export

C_TARGETS = clean standalone milde lib learn_projection evaluate_projection run_kmeans run_LPSR \
	octave oct_find_w.oct oct_normalize_data.oct oct_project_data.oct oct_predict_data.oct \
	oct_get_projection_measures.oct oct_kmeans.oct #oct_evaluate_model.oct oct_calc_objective.oct
PROFILE_TARGETS = $(patsubst %,profile_%,$(C_TARGETS))

.PHONY: all clean octave libsvm liblinear lua libmcfilter

all: octave libsvm liblinear standalone

$(C_TARGETS):
	mkdir -p $(BINDIR)
	$(MAKE) -C $(BINDIR) $(JOBS) -f../c/Makefile $@

#enable profiling
$(PROFILE_TARGETS):
	mkdir -p $(BINDIR)-profile
	$(MAKE) -C $(BINDIR)-profile CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" -f../c/Makefile $(patsubst profile_%,%,$@) 

# lua and libmcfilter targets do not involve octave (just c++ and milde_core's lua)
lua: libmcfilter | libmcfilter
	$(MAKE) -C lua-api $(JOBS)
libmcfilter:
	$(MAKE) -C $(BINDIR) $(JOBS) -f../c/Makefile lib

#standalone: 
#	$(MAKE) -C $(BINDIR) -f../c/Makefile $(MAKECMDGOALS)
#octave:
#	$(MAKE) -C $(BINDIR) -f../octave/Makefile

libsvm:
	cd libsvm-3.17/matlab && make OCTAVE_CFLAGS="$(LIBSVM_OCTAVE_CFLAGS)" octave

liblinear:
	cd liblinear-1.94/matlab && make OCTAVE_CFLAGS="$(LIBSVM_OCTAVE_CFLAGS)" octave


libsvm_clean: 
	cd liblinear-1.94/matlab && make clean
	cd libsvm-3.17/matlab && make clean
#	$(MAKE) -C $(BINDIR) -f../c/Makefile clean
#	$(MAKE) -C lua-api clean
