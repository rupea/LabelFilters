THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
MAKEFILE_DIR = $(abspath $(dir $(THIS_MAKEFILE)))

CXX ?= g++

BOOSTDIR ?= $(HOME)/local
BOOSTINC ?= -I$(BOOSTDIR)/include
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK ?= -L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 

CROARINGDIR ?= ../../CRoaring
CROARINGINC ?= -I$(CROARINGDIR)
CROARINGLIB ?= $(CROARINGDIR)
CROARINGLINK:=-L$(CROARINGLIB) -Wl,-rpath,$(CROARINGLIB) 


OPENMP ?= -fopenmp -DMCTHREADS=1
# for valgrind debug, compile without openmp... then can see "0 errors from 0 contexts"
# 1. no openmp anywhere
#OPENMP:= -Wno-error=unknown-pragmas -Wno-error=unused-variable
# 2. allow Eigen/blas threads, but no MCFilter source code threads
#OPENMP += -DMCTHREADS=0
# allow STL experimental threads?


# testing...
#MKLROOT:=/opt/intel/compilers_and_libraries/linux/mkl/
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -I$(MAKEFILE_DIR)

CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG $(BOOSTINC) $(CROARINGINC) -std=c++11 -pipe -ggdb3 -fPIC

LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(CROARINGLINK)
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = $(OPENMP)
PROFILEDIR = $(HOME)/local
PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP) -std=c++11 -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR) $(CROARINGINC)

PROFILE_CPPFLAGS+=-DMCTHREADS=1

#PROFILE_CPPFLAGS += -DEIGEN_DONT_PARALLELIZE
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

VPATH = ../c


all: lib mcsolve mcproj

%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -c $<


.PHONY: check-headers chk-headers lib libs-only clean realclean hdr-stamp lib-clean
DBGFLAGS := $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS))
DBGFLAGS := $(DBGFLAGS) -ggdb3 -Wall -Wno-sign-compare # -Werror
# for better tracing...
# DBGFLAGS += -O1

%-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -c $<
%.lo: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -fPIC -c $<
%-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -c $<

# overrides to avoid annoying Eigen warnings when 'setFromTriplets' is used
PredictionSet-dbg.o utils-dbg.o : %-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -Wno-sign-compare -c $<
PredictionSet-dbg.lo utils-dbg.lo: %-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -Wno-sign-compare -c $<

/tmp/%.chk: %
	@$(CXX) -c $* $(CPPFLAGS) -o /dev/null && { echo 'OK:   $*'; touch $@; } || { echo 'ERROR: $*'; rm -f $@; }

CHECK_HEADERS := mcfilter.h mcfilter.hh mcsolver.h mcsolver.hh mcsoln.h mcxydata.h mcxydata_detail.h mcxydata_detail.hh mcupdate.hh objective.h mcsolver_detail.hh mcsolver_init.hh boolmatrix.h typedefs.h constants.h mutexlock.h filter.h KMeans.h parameter.h parameter-args.h mclearnFilter.h mcprojector.h linearModel.h linearModel.hh linearModel_detail.h linearModel_detail.hh mclinearClass.h PredictionSet.h printing.h printing.hh utils.h WeightVector.h LPSR.h

# faster version (can run in parallel)
check-headers chk-headers:
	rm -f /tmp/*.chk
	$(MAKE) -f$(THIS_MAKEFILE) -j10 $(patsubst %,/tmp/%.chk,$(CHECK_HEADERS))

LIB_CPPFILES:= mcfilter.cpp mcpredict.cpp mcsolver.cpp mcsoln.cpp mcxydata.cpp filter.cpp PredictionSet.cpp printing.cpp utils.cpp  parameter.cpp parameter-args.cpp mclearnFilter.cpp mcprojector.cpp linearModel.cpp mclinearClass.cpp optimizelu.cpp mcupdate.cpp mcsolver_detail.cpp mcsolver_init.cpp objective.cpp

lib:
	$(MAKE) -f$(THIS_MAKEFILE) -j15 libs-only || { echo -e '\n\n\n\n\n'; $(MAKE) -f$(THIS_MAKEFILE) libs-only; }

lib-clean:
	rm -f libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so *.o *.lo *-dbg.o *-dbg.lo

libs-only: libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so


# Create a library of C++ code for the main functionality in MCFilter
libmcfilter.a: $(patsubst %.cpp,%.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter.so: $(patsubst %.cpp,%.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^ $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring
libmcfilter-dbg.so: $(patsubst %.cpp,%-dbg.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^ $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring
# special library versions
libmcfilter-prof.a: $(patsubst %.cpp,%-prof.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^

%-prof.o: %.cpp
	$(CXX) -o $@ $(PROFILE_CPPFLAGS) -c $<

mcsolve: mcsolve.cpp libmcfilter.so
	$(CXX) -o $@ $(CPPFLAGS) -I$(MAKEFILE_DIR) -Wl,-rpath,$(shell pwd) $^ $(LDFLAGS)  
mcsolve-dbg: mcsolve.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a $(CROARINGLIB)/libroaring.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^

mcproj: mcproj.cpp libmcfilter.so
	$(CXX) -o $@ $(CPPFLAGS) -I$(MAKEFILE_DIR) -Wl,-rpath,$(shell pwd) $^ $(LDFLAGS)  
mcproj-dbg: mcproj.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a $(CROARINGLIB)/libroaring.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) -fpic -L$(HOME)/local/lib $^
mcproj-prof: mcproj.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a $(CROARINGLIB)/libroaring.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) -fpic -L$(HOME)/local/lib $^ -lprofiler -lunwind -llzma

clean: 
	rm -f *.o *.lo *-dbg.o *-dbg.lo *-prof.o libmcfilter*.a libmcfilter*.so \
		mcsolve mcproj

realclean: clean
	rm -f *.soln *.repo *.profile
