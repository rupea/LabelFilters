THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
MAKEFILE_DIR := $(abspath $(dir $(THIS_MAKEFILE)))

# CRoaring is included as a git submodule. Change this to use your own CRoaring install
CROARINGDIR ?= $(MAKEFILE_DIR)/../../CRoaring
CROARINGINC ?= -I$(CROARINGDIR)
CROARINGLIB ?= $(CROARINGDIR)
CROARINGLINK ?= -L$(CROARINGLIB) -Wl,-rpath,$(CROARINGLIB) 

# Eigen is included as a git submodule. Change this ig you want to use your own Eigen install
EIGENDIR ?= $(MAKEFILE_DIR)/../../Eigen
EIGENINC ?= -I$(EIGENDIR)/Eigen

# using boost program options package. Change this to your instalation of boost
BOOSTDIR ?= $(HOME)/local
BOOSTINC ?= -I$(BOOSTDIR)/include
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK ?= -L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 

CXX ?= g++
CC ?= gcc

CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG $(BOOSTINC) $(CROARINGINC) $(EIGENINC) -std=c++11 -pipe 
LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring


VPATH = $(MAKEFILE_DIR) #../c


LIB_CPPFILES:= mcfilter.cpp mcsolver.cpp mcsoln.cpp mcxydata.cpp filter.cpp PredictionSet.cpp printing.cpp utils.cpp  parameter.cpp parameter-args.cpp mclearnFilter.cpp mcprojector.cpp linearModel.cpp mclinearClass.cpp optimizelu.cpp mcupdate.cpp mcsolver_detail.cpp mcsolver_init.cpp objective.cpp

SRCS:= $(LIB_CPPFILES) mcsolve.cpp mcproj.cpp 
DEP:= $(SRCS:%.cpp=%.d)
DEP += $(SRCS:%.cpp=%.ld)

-include $(DEP)

.PHONY: lib lib-clean libs-only clean realclean lib-clean depend .depend


all: lib mcsolve mcproj

%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -MMD -MF $*.d -c $<

%.lo: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -MMD -MF $*.ld -fPIC -c $<

# overrides to avoid annoying Eigen warnings when 'setFromTriplets' is used
#PredictionSet-dbg.o utils-dbg.o : %-dbg.o: %.cpp
#	$(CXX) -o $@ $(DBGFLAGS) -Wno-sign-compare -c $<
#PredictionSet-dbg.lo utils-dbg.lo: %-dbg.lo: %.cpp
#	$(CXX) -o $@ $(DBGFLAGS) -fPIC -Wno-sign-compare -c $<


lib:
	$(MAKE) -f$(THIS_MAKEFILE) -j15 libs-only || { echo -e '\n\n\n\n\n'; $(MAKE) -f$(THIS_MAKEFILE) libs-only; }

lib-clean:
	rm -f libmcfilter.a libmcfilter.so *.o *.lo *.d *.ld

libs-only: libmcfilter.a libmcfilter.so


# Create a library of C++ code for the main functionality in MCFilter
libmcfilter.a: $(patsubst %.cpp,%.o,$(LIB_CPPFILES))
	$(AR) -Uvru $@ $^
libmcfilter.so: $(patsubst %.cpp,%.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^ $(BOOSTLINK) -lboost_program_options $(CROARINGLINK) -lroaring

mcsolve: mcsolve.cpp libmcfilter.so
	$(CXX) -o $@ $(CPPFLAGS) -I$(MAKEFILE_DIR) -Wl,-rpath,$(shell pwd) $^ $(LDFLAGS)  

mcproj: mcproj.cpp libmcfilter.so
	$(CXX) -o $@ $(CPPFLAGS) -I$(MAKEFILE_DIR) -Wl,-rpath,$(shell pwd) $^ $(LDFLAGS)  

clean: lib-clean
	rm -f mcsolve mcproj
