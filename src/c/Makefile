CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
OPENMP ?= -fopenmp
CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) 
CPPFLAGS += -I$(BOOSTDIR) -std=c++11
CPPFLAGS += -fopenmp
CPPFLAGS += -pipe
# testing...
MKLROOT:=/opt/intel/composer_xe_2015.3.187/mkl
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -I.

OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
OCTAVE_CFLAGS ?= -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave

LDFLAGS ?= -L$(BOOSTDIR)/lib -lboost_program_options
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = -fopenmp
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(PROFILE_OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

all: learn_projection evaluate_projection run_LPSR
profile_learnproj: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" learn_projection

profile_evaluate: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" evaluate_projection

learn_projection: learn_projection.o find_w.o printing.o EigenOctave.o utils.o parameter.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o learn_projection $^

evaluate_projection: evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o evaluate_projection evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o

run_kmeans: run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o run_kmeans run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o

run_LPSR: run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o PredictionSet.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o run_LPSR run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o PredictionSet.o

profile_kmeans: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" run_kmeans

%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<

## extra stuff ######################### without octave ###################
DBGFLAGS := $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS))
DBGFLAGS := $(DBGFLAGS) -ggdb3 -Wall -Wno-error=sign-compare -Werror
%-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -c $<
%.lo: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -fPIC -c $<
%-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -c $<
settings:
	echo "boost wildcard check <$(wildcard /opt/boost-1.60)>"
	echo "BOOSTDIR       = $(BOOSTDIR)"
check-headers: boolmatrix.h EigenIO.h evaluate.h filter.h find_w.h KMeans.h parameter.h predict.h PredictionSet.h printing.h utils.h WeightVector.h
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking standalone header compilation"
lib: libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so demo-proj
LIB_CPPFILES:=find_w.cpp filter.cpp PredictionSet.cpp printing.cpp utils.cpp EigenIO.cpp parameter.cpp
# Create a library of octave-free C++ code for the main functionality in MCFilter
#   (find_w, filter, kmeans, LPSR)
#   This should aid in producing an interface in lua [I hope]
libmcfilter.a: $(patsubst %.cpp,%.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter.so: $(patsubst %.cpp,%.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^
libmcfilter-dbg.so: $(patsubst %.cpp,%-dbg.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^
# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
demo-proj: | libmcfilter.a
demo-proj: erik/demo-proj.cpp libmcfilter.a
	$(CXX) -o $@ $(CPPFLAGS) -I. $^ $(LDFLAGS)
	@#/usr/bin/time -v ./demo-proj 2>&1 | tee demo-proj.log # 917 seconds
# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
demo-proj-dbg: | libmcfilter-dbg.a
demo-proj-dbg: erik/demo-proj.cpp libmcfilter-dbg.a
	$(CXX) -o $@ $(CPPFLAGS) -I. $^ $(LDFLAGS)
# The following test throws if the solution is not sufficiently close to the correct answer.
# Note: inside eigen, there are signed vs. unsigned issues (error=sign-compare)
demo-quick: libmcfilter.a
demo-quick: erik/demo-quick.cpp libmcfilter.a
	$(CXX) -o $@ $(DBGFLAGS) -I. $^ $(LDFLAGS)

clean: 
	rm -f *.o *.lo learn_projection evaluate_projection run_kmeans demo-quick demo-proj demo-proj-dbg libmcfilter*.a libmcfilter*.so
