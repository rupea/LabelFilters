CXX ?= g++
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
OPENMP ?= -fopenmp
CPPFLAGS ?= -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) 
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
OCTAVE_CFLAGS ?= -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave

LDFLAGS ?= -L$(BOOSTDIR)/lib -lboost_program_options
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = -L$(PROFILEDIR)/lib -lprofiler

all: learn_projection evaluate_projection

profile: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" main main_evaluate

learn_projection: learn_projection.o find_w.o printing.o EigenOctave.o utils.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o learn_projection learn_projection.o EigenOctave.o find_w.o printing.o utils.o

evaluate_projection: evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o evaluate_projection evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o

.cpp.o:
	$(CXX) $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o learn_projection evaluate_projection
