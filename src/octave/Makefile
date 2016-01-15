CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif

ifneq ($(wildcard $(HOME)/Programs/octave-3.8.2),)
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
endif
ifneq ($(wildcard /usr/include/octave-3.8.2),)
OCTAVEDIR ?= /usr
endif
OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
HDF5_INC ?= -I/usr/include/hdf5/serial
OCTAVE_CFLAGS ?= -O3 -fPIC -fopenmp -std=c++11 -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave $(HDF5_INC) -DNDEBUG -DEIGEN_NO_DEBUG  -I$(BOOSTDIR)
OCTAVE_LDFLAGS ?= -L../c

VPATH = ../c
# ??? but need absolute dir for rpath --- OCTAVE_LDFLAGS = -WL,-rpath,$(VPATH) $(OCTAVE_LDFLAGS)

SRCS	= find_w.cpp EigenOctave.cpp 
OBJS	= ${SRCS:.c=.o}

OCTFILES = oct_find_w.oct oct_normalize_data.oct

.PHONY: clean all octave-objects octave-executables

.SUFFIXES:
.SUFFIXES: .cpp .o

all: octave-objects octave-executables

octave-objects:
	$(MAKE) -j10 -C../c -fMakefile
	$(MAKE) -j10 -C../c -fMakefile.octave

octave-executables: oct_find_w oct_project_data oct_predict_data oct_evaluate_model oct_normalize_data oct_evaluate_predictions

oct_find_w: oct_find_w.o find_w.o EigenOctave.o printing.o utils.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_normalize_data: oct_normalize_data.o normalize.o EigenOctave.o printing.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_project_data: oct_project_data.o EigenOctave.o filter.o utils.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_predict_data: oct_predict_data.o EigenOctave.o filter.o utils.o PredictionSet.o predict.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_evaluate_model: oct_evaluate_model.o EigenOctave.o filter.o utils.o PredictionSet.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_evaluate_predictions: oct_evaluate_predictions.o EigenOctave.o utils.o PredictionSet.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_get_projection_measures: oct_get_projection_measures.o EigenOctave.o utils.o filter.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_calc_objective: find_w.o EigenOctave.o oct_calc_objective.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_kmeans: utils.o printing.o EigenOctave.o normalize.o oct_kmeans.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

.cpp.o:
	$(CXX) $(OCTAVE_CFLAGS) -c $<

clean: 
	rm -f *.o *.oct
