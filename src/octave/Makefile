OCTAVEDIR ?= $(HOME)/Programs/octave-3.6.4/
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile

OCTFILES = oct_find_w.oct oct_normalize_data.oct oct_calc_objective.oct


all: octave

octave: $(OCTFILES)
	$(OCTAVE_MEX) ../c/oct_find_w.cpp
	$(OCTAVE_MEX) ../c/oct_normalize_data.cpp
	$(OCTAVE_MEX) ../c/oct_calc_objective.cpp

.PHONY: clean

clean: 
	rm -f *.o *.oct
