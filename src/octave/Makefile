CXX ?= g++
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile
OCTAVE_CFLAGS ?= -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
OCTAVE_LDFLAGS ?= 

PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O2 -DPROFILE -I$(PROFILEDIR)/include -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
PROFILE_LDFLAGS = -L$(PROFILEDIR)/lib -lprofiler


VPATH = ../c

SRCS	= find_w.cpp EigenOctave.cpp 
OBJS	= ${SRCS:.c=.o}

OCTFILES = oct_find_w.oct oct_normalize_data.oct

.PHONY: clean all octave

.SUFFIXES:
.SUFFIXES: .cpp .o

all: octave

profile: 
	@make OCTAVE_CFLAGS="$(PROFILE_CPPFLAGS)" OCTAVE_LDFLAGS="$(PROFILE_LDFLAGS)" octave


octave: oct_find_w oct_normalize_data


oct_find_w: find_w.o EigenOctave.o printing.o boolmatrix.o ../c/oct_find_w.cpp
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) ../c/oct_find_w.cpp find_w.o EigenOctave.o printing.o boolmatrix.o

oct_normalize_data: normalize.o EigenOctave.o printing.o ../c/oct_normalize_data.cpp
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) ../c/oct_normalize_data.cpp normalize.o EigenOctave.o printing.o 

oct_calc_objective: find_w.o EigenOctave.o ../c/oct_calc_objective.cpp
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS)  ../c/oct_calc_objective.cpp find_w.o EigenOctave.o

.cpp.o:
	$(CXX) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o *.oct
