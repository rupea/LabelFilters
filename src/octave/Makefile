CXX ?= g++
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile
OCTAVE_CFLAGS ?= -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave -DNDEBUG -DEIGEN_NO_DEBUG  -I$(BOOSTDIR)
OCTAVE_LDFLAGS ?= 

VPATH = ../c

SRCS	= find_w.cpp EigenOctave.cpp 
OBJS	= ${SRCS:.c=.o}

OCTFILES = oct_find_w.oct oct_normalize_data.oct

.PHONY: clean all octave

.SUFFIXES:
.SUFFIXES: .cpp .o

all: octave

octave: oct_find_w oct_normalize_data


oct_find_w: find_w.o EigenOctave.o printing.o oct_find_w.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) oct_find_w.o find_w.o EigenOctave.o printing.o 

oct_normalize_data: normalize.o EigenOctave.o printing.o oct_normalize_data.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) oct_normalize_data.o normalize.o EigenOctave.o printing.o 

oct_calc_objective: find_w.o EigenOctave.o oct_calc_objective.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS)  oct_calc_objective.o find_w.o EigenOctave.o

.cpp.o:
	$(CXX) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o *.oct
