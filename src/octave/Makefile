CXX ?= g++
OCTAVEDIR ?= $(HOME)/Programs/octave-3.6.4
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile
OCTAVE_CFLAGS ?= -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.6.4 -I..

VPATH = ../c

SRCS	= find_w.cpp EigenOctave.cpp 
OBJS	= ${SRCS:.c=.o}

OCTFILES = oct_find_w.oct oct_normalize_data.oct oct_calc_objective.oct

.PHONY: clean all octave

.SUFFIXES:
.SUFFIXES: .cpp .o

all: octave

octave: oct_find_w oct_normalize_data oct_calc_objective


oct_find_w: find_w.o EigenOctave.o printing.o ../c/oct_find_w.cpp
	$(OCTAVE_MEX) ../c/oct_find_w.cpp find_w.o EigenOctave.o printing.o

oct_normalize_data: normalize.o EigenOctave.o printing.o ../c/oct_normalize_data.cpp
	$(OCTAVE_MEX) ../c/oct_normalize_data.cpp normalize.o EigenOctave.o printing.o 

oct_calc_objective: find_w.o EigenOctave.o ../c/oct_calc_objective.cpp
	$(OCTAVE_MEX)  ../c/oct_calc_objective.cpp find_w.o EigenOctave.o

.cpp.o:
	$(CXX) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o *.oct
