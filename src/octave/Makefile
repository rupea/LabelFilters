OCTAVEDIR ?= ~/Programs/octave-3.6.4/
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile

OCTFILES = oct_find_w.oct oct_normalize_data.oct oct_calc_objective.oct


.PHONY: all octave clean

all: octave

octave:
	$(OCTAVE_MEX) ../c/oct_find_w.cpp
	$(OCTAVE_MEX) ../c/oct_normalize_data.cpp
	$(OCTAVE_MEX) ../c/oct_calc_objective.cpp


clean: 
	rm -f *.o *.oct
