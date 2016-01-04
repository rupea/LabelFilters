CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
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
LDFLAGS += -L$(MILDE_DIR)/lib -lmilde_core -lmilde_lua
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = -fopenmp
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(PROFILE_OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

all: libmclua.a libmclua-dbg.a linkme
profile_learnproj: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" learn_projection

profile_evaluate: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" evaluate_projection


%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<
%.i: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(OCTAVE_CFLAGS) -dD -E $<
%-dbg.o: %.cpp
	$(CXX) $(DBGFLAGS) -c $< -o $@
../c/lib%.a:
	cd ../c && $(MAKE) lib$*.a

settings:
	echo "boost wildcard check <$(wildcard /opt/boost-1.60)>"
	echo "BOOSTDIR       = $(BOOSTDIR)"
check-headers:
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking standalone header compilation"

lib: libmclua.a libmclua-dbg.a
LIBMCLUA_FILES:=mclua.o
libmclua.a: ../c/libmcfilter.a
libmclua-dbg.a: ../c/libmcfilter-dbg.a
libmclua.a: $(LIBMCLUA_FILES)
	$(AR) -vru $@ $^
libmclua-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIBMCLUA_FILES))
	$(AR) -vru $@ $^

linkme: linkme.cpp
	$(CXX) -o $@ $(CPPFLAGS) linkme.cpp mclua.cpp $(LDFLAGS)

clean: 
	rm -f *.o linkme libmclua*.a *.i
