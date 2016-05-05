# This Makefile is used under Linux

MATLABDIR ?= /usr/local/matlab
# for Mac
# MATLABDIR ?= /opt/local/matlab

CXX ?= g++
#CXX = g++-4.1
CFLAGS = -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)"
# comment the following line if you use MATLAB on 32-bit computer
MEX_OPTION += -largeArrayDims
#MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

ifneq ($(wildcard $(HOME)/Programs/octave-3.8.2),)
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
OCT_INC:=-I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
endif
ifneq ($(wildcard /usr/include/octave-3.8.2),)
OCTAVEDIR ?= /usr
OCT_INC:=-I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
endif
ifneq ($(wildcard /usr/include/octave-3.8.1),)
OCTAVEDIR = /usr
OCT_INC:=-I$(OCTAVEDIR)/include/octave-3.8.1 -I$(OCTAVEDIR)/include/octave-3.8.1/octave
endif
OCTAVE_MEX = env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile
OCTAVE_MEX_OPTION = --mex
OCTAVE_MEX_EXT = mex
#OCTAVE_CFLAGS ?= -Wall -O3 -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2/octave/ -I..
OCTAVE_CFLAGS ?= -Wall -O3 -fPIC $(OPENMP) -std=c++11 $(OCT_INC) -DNDEBUG -I..

all:	matlab

matlab:	binary

octave:
	@make MEX="$(OCTAVE_MEX)" MEX_OPTION="$(OCTAVE_MEX_OPTION)" \
	MEX_EXT="$(OCTAVE_MEX_EXT)" CFLAGS="$(OCTAVE_CFLAGS)" \
	binary

binary: svmpredict.$(MEX_EXT) svmtrain.$(MEX_EXT) libsvmread.$(MEX_EXT) libsvmwrite.$(MEX_EXT) read_sparse_ml.$(MEX_EXT)

svmpredict.$(MEX_EXT):     svmpredict.c ../svm.h ../svm.o svm_model_matlab.o
	$(MEX) $(MEX_OPTION) svmpredict.c ../svm.o svm_model_matlab.o

svmtrain.$(MEX_EXT):       svmtrain.c ../svm.h ../svm.o svm_model_matlab.o
	$(MEX) $(MEX_OPTION) svmtrain.c ../svm.o svm_model_matlab.o

libsvmread.$(MEX_EXT):	libsvmread.c
	$(MEX) $(MEX_OPTION) libsvmread.c

libsvmwrite.$(MEX_EXT):	libsvmwrite.c
	$(MEX) $(MEX_OPTION) libsvmwrite.c

svm_model_matlab.o:     svm_model_matlab.c ../svm.h
	$(CXX) $(CFLAGS) -c svm_model_matlab.c

read_sparse_ml.$(MEX_EXT): read_sparse_ml.c
	$(MEX) $(MEX_OPTION) read_sparse_ml.c	

../svm.o: ../svm.cpp ../svm.h
	make -C .. svm.o

clean:
	rm -f *~ *.o *.mex* *.obj ../svm.o
