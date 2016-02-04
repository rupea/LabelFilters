CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(notdir $(patsubst %/,%,$(dir $(MKFILE_PATH))))
# the directory above this makefile...
MCSRC_DIR := $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(MKFILE_PATH)))))

OPENMP ?= -fopenmp
# - removed -ffastmath (interferes with isnan for milde compilation)
CPPFLAGS ?= -march=native -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) 
CPPFLAGS += -I$(BOOSTDIR) -std=c++11
CPPFLAGS += -fopenmp
CPPFLAGS += -I../c -I$(MILDE_DIR)/include/milde_lua -I$(MILDE_DIR)/src/lib
CPPFLAGS += -Wno-attributes
# testing...
MKLROOT:=/opt/intel/composer_xe_2015.3.187/mkl
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -I.
DBGFLAGS := $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS)) -ggdb3 -Wall -Werror

LDFLAGS ?= -L$(BOOSTDIR)/lib -lboost_program_options
LDFLAGS += -L$(MILDE_DIR)/lib -Wl,--no-as-needed -lmilde_core -Wl,--no-as-needed -lmilde_lua
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = -fopenmp
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(PROFILE_OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

.PHONY: lib
all: lib linkme
profile_learnproj: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" learn_projection

profile_evaluate: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" evaluate_projection


%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -c $<
%.i: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -dD -E $<
%-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -c $<
%.lo: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -fPIC -c $<
%-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -c $<
../c/lib%.a:
	cd ../c && $(MAKE) lib$*.a

settings:
	echo "boost wildcard check <$(wildcard /opt/boost-1.60)>"
	echo "BOOSTDIR       = $(BOOSTDIR)"
	echo "MKFILE_PATH    = $(MKFILE_PATH)"
	echo "CURRENT_DIR    = $(CURRENT_DIR)"
	echo "MCSRC_DIR      = $(MCSRC_DIR)"
check-headers:
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking standalone header compilation"

#lib: libmclua.a libmclua-dbg.a
lib: libmclua.so libmclua-dbg.so
#LIBMCLUA_FILES:=mclua.o
#libmclua.a: ../c/libmcfilter.a
#libmclua-dbg.a: ../c/libmcfilter-dbg.a
#libmclua.a: $(LIBMCLUA_FILES)
#	$(AR) -vru $@ $^
#libmclua-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIBMCLUA_FILES))
#	$(AR) -vru $@ $^
LIBMCLUA_SRC:=mclua.cpp

# To ease linking, wire the mcfilter library & path into the lua library
# (so that require('libmclua') will auto-load libmcfilter.so without doing
#  complicated song and dance in lua)
# rpath setting stored within libmclua*.so should be an absolute path
#  TODO: eventually, perhaps to some milde install directory?
DEP_LIB_DIR:=-Wl,-rpath,$(MCSRC_DIR)/c -L../c

# Note:
#       mcparm.so can link to EITHER libmclua.so OR libmclua-dbg.so
# Then if the lua_CPATH includes ".", you can require 'mcparm', which will:
# 1) load the lua api from mcparms.so and
# 2) link to the libmcfilter compiled at corresponding debug level
libmclua.so: $(patsubst %.cpp,%.lo,$(LIBMCLUA_SRC))
	$(CXX) -o $@ -shared $^ $(DEP_LIB_DIR) -lmcfilter
	# rm -f mcparm.so && ln -sf libmclua-dbg.so mcparm.so
libmclua-dbg.so: $(patsubst %.cpp,%-dbg.lo,$(LIBMCLUA_SRC))
	$(CXX) -o $@ -shared $^ $(DEP_LIB_DIR) -lmcfilter-dbg
	rm -f mcparm.so && ln -sf libmclua-dbg.so mcparm.so

# links, but don't know how to "run" lua from inside C++ code yet...
#   For now, you can test as   make clean && make && lua_cpp parms.lua
linkme: linkme.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -O0 linkme.cpp -L. -Wl,-rpath,. -Wl,-rpath,../c -lmclua-dbg -L../c -lmcfilter $(LDFLAGS)

clean: 
	rm -f *.o *.lo *.so linkme libmclua*.a *.i
