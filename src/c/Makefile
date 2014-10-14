CXX ?= g++
CPPFLAGS ?= -g -O3
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
OCTAVE_CFLAGS ?= -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave

LDFLAGS ?=
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O2 -DPROFILE -I$(PROFILEDIR)/include 
PROFILE_LDFLAGS = -L$(PROFILEDIR)/lib -lprofiler

all: main

profile: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" main

main: main.o find_w.o printing.o EigenOctave.o boolmatrix.o
	$(OCTAVE_MEX) $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) -o main main.o boolmatrix.o EigenOctave.o find_w.o printing.o 

.cpp.o:
	$(CXX) $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o main find_w.profile
