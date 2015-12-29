CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
OPENMP ?= -fopenmp
CPPFLAGS ?= -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) 
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
OCTAVE_CFLAGS ?= -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave

LDFLAGS ?= -L$(BOOSTDIR)/lib -lboost_program_options
PROFILE_OPENMP = -fopenmp
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(PROFILE_OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

all: learn_projection evaluate_projection run_LPSR
lib: settings libmcfilter.a
settings:
	echo "boost wildcard check <$(wildcard /opt/boost-1.60)>"
	echo "BOOSTDIR       = $(BOOSTDIR)"
# Create a library of octave-free C++ code for the main functionality in MCFilter
#   (find_w, filter, kmeans, LPSR)
#   This should aid in producing an interface in lua [I hope]
libmcfilter.a: find_w.o filter.o PredictionSet.o printing.o utils.o
	$(AR) -vru $@ $^

profile_learnproj: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" learn_projection

profile_evaluate: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" evaluate_projection

learn_projection: learn_projection.o find_w.o printing.o EigenOctave.o utils.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o learn_projection learn_projection.o EigenOctave.o find_w.o printing.o utils.o

evaluate_projection: evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o evaluate_projection evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o

run_kmeans: run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o run_kmeans run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o

run_LPSR: run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o PredictionSet.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o run_LPSR run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o PredictionSet.o

profile_kmeans: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" run_kmeans

.cpp.o:
	$(CXX) $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o learn_projection evaluate_projection run_kmeans
