CXX ?= g++
ifneq ($(wildcard /opt/boost-1.60),)
BOOSTDIR ?= /opt/boost-1.60/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
OPENMP ?= -fopenmp
CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) 
CPPFLAGS += -I$(BOOSTDIR) -std=c++11
CPPFLAGS += -fopenmp
CPPFLAGS += -pipe
# testing...
MKLROOT:=/opt/intel/composer_xe_2015.3.187/mkl
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -I.

ifneq ($(wildcard $(HOME)/Programs/octave-3.8.2),)
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
endif
ifneq ($(wildcard /usr/include/octave-3.8.2),)
OCTAVEDIR ?= /usr
endif

OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
HDF5_INC ?= -I/usr/include/hdf5/serial
OCTAVE_CFLAGS ?= -fPIC -I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave $(HDF5_INC)

LDFLAGS ?= -L$(BOOSTDIR)/../lib -lboost_program_options
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = -fopenmp
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O3 $(PROFILE_OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

all: learn_projection evaluate_projection run_LPSR
profile_learnproj: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" learn_projection

profile_evaluate: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" evaluate_projection

learn_projection: learn_projection.o find_w.o find_w_detail.o mcsolver.o printing.o EigenOctave.o utils.o parameter.o
	$(OCTAVE_MEX) -o $@ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) $*

evaluate_projection: evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o evaluate.o
	$(OCTAVE_MEX) -o $@ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) $*

run_kmeans: run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o
	$(OCTAVE_MEX) -o $@ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) $*

run_LPSR: run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o evaluate.o PredictionSet.o
	$(OCTAVE_MEX) -o $@ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE) $*

profile_kmeans: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" run_kmeans

%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) $(OCTAVE_CFLAGS) -c $<


## extra stuff ######################### without octave ###################
.PHONY: settings check-headers chk-headers lib libs-only lib test long long-compare do-tests
DBGFLAGS := $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS))
DBGFLAGS := $(DBGFLAGS) -ggdb3 -Wall -Wno-error=sign-compare -Werror
%-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -c $<
%.lo: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -fPIC -c $<
%-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -c $<
# overrides to avoid annoying Eigen warnings when 'setFromTriplets' is used
PredictionSet-dbg.o utils-dbg.o : %-dbg.o: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -Wno-sign-compare -c $<
PredictionSet-dbg.lo utils-dbg.lo: %-dbg.lo: %.cpp
	$(CXX) -o $@ $(DBGFLAGS) -fPIC -Wno-sign-compare -c $<
settings:
	echo "boost wildcard check <$(wildcard /opt/boost-1.60)>"
	echo "BOOSTDIR       = $(BOOSTDIR)"
/tmp/%.chk: %
	@$(CXX) -c $* $(CPPFLAGS) -o /dev/null && { echo 'OK:   $*'; touch $@; } || { echo 'ERROR: $*'; rm -f $@; }
CHECK_HEADERS := mcsolver.h mcsolver.hh boolmatrix.h EigenIO.h typedefs.h constants.h normalize.h evaluate.h evaluate.hh mutexlock.h filter.h find_w.h KMeans.h parameter.h predict.h predict.hh PredictionSet.h printing.h printing.hh utils.h WeightVector.h find_w_detail.h find_w_detail.hh find_w.hh LPSR.h 
check-headers: $(CHECK_HEADERS)
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking standalone header compilation"
# faster version (can run in parallel)
chk-headers:
	rm -f /tmp/*.chk
	$(MAKE) -j10 $(patsubst %,/tmp/%.chk,$(CHECK_HEADERS))
check-oct-headers: EigenOctave.h
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) $(OCTAVE_CFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking octave-related standalone header compilation"
lib: libs-only demo-proj demo-quick
test: lib demo-printing demo-io do-tests mcgen
long: clean
	{ $(MAKE) -j20 demo-mcsolver && ./demo-mcsolver 2>&1 | tee dmc.log | (head -n5; echo "^^^... ... ...vvv"; tail -n20); } \
	&& $(MAKE) -j20 do-tests \
	&& $(MAKE) -j20 demo-proj demo-proj-dbg demo-proj2-dbg \
	&& { if [ -f dpa.log ]; then $(MAKE) dpa.log; else tail -n5 dpa.log; fi; } \
	&& $(MAKE) dpb.log
	rm -f dpa1.log dpb1.log dpa2.log dpb2.log dpa3.log dpb3.log dpa4.log dpb4.log dpa5.log dpb5.log dpa6.log dpb6.log
	$(MAKE) -j6 dpa1.log dpb1.log dpa2.log dpb2.log dpa3.log dpb3.log dpa4.log dpb4.log dpa5.log dpb5.log dpa6.log dpb6.log > erik/mk-long.log 2>&1
	$(MAKE) long-compare
long-compare:
	{ for n in "" 1 2 3 4 5 6; do \
	  a=`tail dpa$$n.log | grep '^obj' | tail -n1`; \
	  b=`tail dpb$$n.log | grep '^obj' | tail -n1`; \
	  echo "dpa$$n: $$a     dpb$$n: $$b"; \
	done; } \
	&& echo 'OK: make long'
dpa.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa1.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 1 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb1.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 1 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa2.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 2 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb2.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 2 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa3.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 3 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb3.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 3 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa4.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 4 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb4.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 4 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa5.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 5 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb5.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 5 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa6.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 6 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb6.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 6 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
do-tests: | lib demo-printing demo-io demo-quick demo-mcsolver
	./demo-quick > demo-quick.log 2>&1
	./demo-printing > demo-printing.log 2>&1
	./demo-io > demo-io.log 2>&1
	./demo-mcsolver > demo-mcsolver.log 2>&1
	# All the simple tests (p0) can converge, for both original (o) and mcsolver (m,default)
	@nfails=0; ntests=0; \
	for t in 0 1 2 3 4 5 6; do  \
	  { ./demo-mcsolver op0$$t > erik/log-op0$$t.log 2>&1 || { nfails=$$(($$nfails+1)); echo "ERROR in demo-mcsolver op0$$t"; false; } ; } &\
	  { ./demo-mcsolver mp0$$t > erik/log-mp0$$t.log 2>&1 || { nfails=$$(($$nfails+1)); echo "ERROR in demo-mcsolver mp0$$t"; false; } ; } &\
	  wait; \
	  ntests=$$(($$ntests+1)); \
	done; \
	if [ "$$nfails" = "0" ]; then echo "                do-tests *** PASSED $$ntests TESTS ***"; \
	else                          echo "                do-tests *** FAILED $$nfails TESTS ***"; false; fi
libs-only: libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so
LIB_CPPFILES:= mcsolver.cpp find_w.cpp find_w_detail.cpp filter.cpp PredictionSet.cpp printing.cpp utils.cpp EigenIO.cpp parameter.cpp evaluate.cpp predict.cpp
# Create a library of octave-free C++ code for the main functionality in MCFilter
#   (find_w, filter, kmeans, LPSR)
#   This should aid in producing an interface in lua [I hope]
libmcfilter.a: $(patsubst %.cpp,%.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter.so: $(patsubst %.cpp,%.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^
libmcfilter-dbg.so: $(patsubst %.cpp,%-dbg.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^
# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
demo-proj: | libmcfilter.a
demo-proj: erik/demo-proj.cpp libmcfilter.a
	$(CXX) -o $@ $(CPPFLAGS) -I. $^ $(LDFLAGS)
	@#/usr/bin/time -v ./demo-proj 2>&1 | tee demo-proj.log # 917 seconds
# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
demo-proj-dbg: | libmcfilter-dbg.a
demo-proj-dbg: erik/demo-proj.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -DUSE_MCSOLVER=0 $(CPPFLAGS) -I. $^ $(LDFLAGS)
demo-proj2-dbg: | libmcfilter-dbg.a
demo-proj2-dbg: erik/demo-proj.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -DUSE_MCSOLVER=1 $(CPPFLAGS) -I. $^ $(LDFLAGS)
# The following test throws if the solution is not sufficiently close to the correct answer.
# Note: inside eigen, there are signed vs. unsigned issues (error=sign-compare)
demo-quick: | libmcfilter-dbg.a
demo-quick: erik/demo-quick.cpp libmcfilter-dbg.a
	$(CXX) -o $@ $(DBGFLAGS) -I. $^ $(LDFLAGS)
demo-mcsolver: | libmcfilter-dbg.a
demo-mcsolver: erik/demo-mcsolver.cpp libmcfilter-dbg.a
	$(CXX) -o $@ $(DBGFLAGS) -I. $^ $(LDFLAGS)
demo-printing: erik/demo-printing.cpp
	$(CXX) -o $@ $(DBGFLAGS) -I. $^

demo-io: erik/demo-io.cpp libmcfilter-dbg.a
	$(CXX) -o $@ $(DBGFLAGS) -I. $^
#static linkage : 2.6M executable
#mcgen: erik/mcgen.cpp libmcfilter-dbg.a $(BOOSTDIR)/../lib/libboost_program_options.a
mcgen: erik/mcgen.cpp $(BOOSTDIR)/../lib/libboost_program_options.a
	$(CXX) -o $@ $(DBGFLAGS) -I. $^
#shared linkage : 2.1M executable.. Not much savings.
#mcgen: erik/mcgen.cpp | libmcfilter-dbg.so
#	$(CXX) -o $@ -fPIC $(DBGFLAGS) -I. $< -Wl,-Bdynamic -L. -lmcfilter-dbg -L$(BOOSTDIR)/../lib -lboost_program_options -Wl,-rpath=$(BOOSTDIR)/../lib -Wl,-rpath=.

clean: 
	rm -f *.o *.lo learn_projection evaluate_projection run_kmeans demo-quick demo-proj demo-proj-dbg libmcfilter*.a libmcfilter*.so
	rm -f demo-quick demo-printing demo-io demo-quick.log demo-printing.log demo-quick.log run_LPSR demo-proj2-dbg demo-mcsolver demo-proj.native mcgen
	$(MAKE) -f Makefile.octave clean
