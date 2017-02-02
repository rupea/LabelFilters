THIS_MAKEFILE = $(lastword $(MAKEFILE_LIST))
MAKEFILE_DIR = $(abspath $(dir $(THIS_MAKEFILE)))

CXX ?= g++

ifneq ($(wildcard /opt/boost-1.60),)
BOOSTROOT?=/opt/boost-1.60
BOOSTDIR :=$(BOOSTROOT)/include
endif
ifneq ($(wildcard $(HOME)/Programs/boost_1_56_0),)
BOOSTDIR ?= $(HOME)/Programs/boost_1_56_0
endif
BOOSTLIB ?= $(BOOSTDIR)/lib
BOOSTLINK:=-L$(BOOSTLIB) -Wl,-rpath,$(BOOSTLIB) 


OPENMP ?= -fopenmp
# for valgrind debug, compile without openmp... then can see "0 errors from 0 contexts"
# 1. no openmp anywhere
#OPENMP:= -Wno-error=unknown-pragmas -Wno-error=unused-variable
# 2. allow Eigen/blas threads, but no MCFilter source code threads
#OPENMP:= -DMCTHREADS=0
# allow STL experimental threads?


#CPPFLAGS += -fopenmp
# testing...
MKLROOT:=/opt/intel/composer_xe_2015.3.187/mkl
#CPPFLAGS += -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -I$(MAKEFILE_DIR)

ifneq ($(wildcard /usr/include/octave-3.8.2),)
OCTAVEDIR ?= /usr
endif
ifneq ($(wildcard $(HOME)/Programs/octave-3.8.2),)
OCTAVEDIR ?= $(HOME)/Programs/octave-3.8.2
endif
OCT_INC:=-I$(OCTAVEDIR)/include/octave-3.8.2 -I$(OCTAVEDIR)/include/octave-3.8.2/octave
OCT_LINK:= -Wl,-rpath,$(OCTAVEDIR)/lib64 -Wl,-rpath,$(OCTAVEDIR)/lib64/octave/3.8.2

OCTAVE_MEX ?= env CC=$(CXX) $(OCTAVEDIR)/bin/mkoctfile -v -lgomp
OCTAVE_MEX_STANDALONE ?= --link-stand-alone
HDF5_INC ?= -I/usr/include/hdf5/serial
#OCTAVE_CFLAGS ?= -fPIC $(OCT_INC) $(HDF5_INC)

CPPFLAGS ?= -march=native -ffast-math -m64 -O3 $(OPENMP) -DNDEBUG -DEIGEN_NO_DEBUG -I$(BOOSTDIR) -std=c++11 -pipe -ggdb3 -fPIC $(OCT_INC) $(HDF5_INC)

LDFLAGS ?= $(BOOSTLINK) -lboost_program_options $(OCT_LINK)
# testing
#LDFLAGS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a -Wl,--end-group -ldl -lpthread -lm

PROFILE_OPENMP = $(OPENMP)
PROFILEDIR = $(HOME)/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -ggdb3 -O3 $(PROFILE_OPENMP) -std=c++11 -DNDEBUG -DEIGEN_NO_DEBUG -DPROFILE -I$(PROFILEDIR)/include -I$(BOOSTDIR)
PROFILE_CPPFLAGS+=-DMCTHREADS=1
#PROFILE_CPPFLAGS += -DEIGEN_DONT_PARALLELIZE
PROFILE_LDFLAGS = $(LDFLAGS) -L$(PROFILEDIR)/lib -lprofiler 

VPATH = ../c 

#OCTAVE_LDFLAGS = 
#OCTAVE_CFLAFS = 

all: standalone octave

standalone:  learn_projection evaluate_projection run_kmeans naive_bayes #demo-mcsolver mcgen mcgenx mcdumpsoln demo-proj1 erik/mcsolve0 mcsolve run_LPSR

octave: oct_find_w.oct oct_normalize_data.oct oct_project_data.oct oct_predict_data.oct oct_evaluate_predictions.oct oct_get_projection_measures.oct oct_kmeans.oct  #oct_evaluate_model.oct oct_calc_objective.oct


# For MiLDe, remove all "octave"-related targets, and ensure 'lib' target[s] are mode
milde: lib demo-mcsolver mcgen mcgenx mcdumpsoln demo-proj1 erik/mcsolve0 mcsolve | lib


learn_projection: learn_projection.o find_w.o find_w_detail.o optimizelu.o mcsolver.o printing.o EigenOctave.o utils.o parameter.o predict.o filter.o
	$(OCTAVE_MEX) -o $@ $^ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE)

evaluate_projection: evaluate_projection.o EigenOctave.o filter.o utils.o PredictionSet.o EigenIO.o evaluate.o predict.o
	$(OCTAVE_MEX) -o $@ $^ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE)

run_kmeans: run_kmeans.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o
	$(OCTAVE_MEX) -o $@ $^ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE)

#run_LPSR: run_LPSR.o utils.o printing.o EigenOctave.o normalize.o EigenIO.o evaluate.o PredictionSet.o predict.o filter.o
#	$(OCTAVE_MEX) -o $@ $^ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE)

naive_bayes: naive_bayes.o EigenOctave.o utils.o
	$(OCTAVE_MEX) -o $@ $^ $(LDFLAGS) $(OCTAVE_MEX_STANDALONE)


oct_find_w.oct: oct_find_w.o find_w.o EigenOctave.o printing.o utils.o parameter.o find_w_detail.o optimizelu.o mcsolver.o 
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_normalize_data.oct: oct_normalize_data.o normalize.o EigenOctave.o printing.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_project_data.oct: oct_project_data.o EigenOctave.o filter.o utils.o
	$(OCTAVE_MEX)  $(OCTAVE_LDFLAGS) $^

oct_predict_data.oct: oct_predict_data.o EigenOctave.o filter.o utils.o PredictionSet.o predict.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

# this functions is deprecated and will not compile. If needed, oct_evaluate_model.cpp needs to be updated 
# to incorporate changes in the code
# oct_evaluate_model.oct: oct_evaluate_model.o EigenOctave.o filter.o utils.o PredictionSet.o
#	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_evaluate_predictions.oct: oct_evaluate_predictions.o EigenOctave.o utils.o PredictionSet.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_get_projection_measures.oct: oct_get_projection_measures.o EigenOctave.o utils.o filter.o
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

# this functions is deprecated and will not compile. 
# to incorporate changes in the code
#oct_calc_objective.oct: find_w.o EigenOctave.o oct_calc_objective.o
#	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^

oct_kmeans.oct: oct_kmeans.o utils.o printing.o EigenOctave.o normalize.o 
	$(OCTAVE_MEX) $(OCTAVE_LDFLAGS) $^


%.o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -c $<


## extra stuff ######################### without octave ###################
.PHONY: settings check-headers chk-headers lib libs-only lib test long long-compare do-tests do-vg tail-vg clean realclean hdr-stamp lib-clean
DBGFLAGS := $(filter-out -DNDEBUG -DEIGEN_NO_DEBUG,$(CPPFLAGS))
DBGFLAGS := $(DBGFLAGS) -ggdb3 -Wall -Wno-sign-compare -Werror
# for better tracing...
DBGFLAGS += -O1
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
	echo "BOOSTLIB       = $(BOOSTLIB)"
/tmp/%.chk: %
	@$(CXX) -c $* $(CPPFLAGS) -o /dev/null && { echo 'OK:   $*'; touch $@; } || { echo 'ERROR: $*'; rm -f $@; }
CHECK_HEADERS := mcpredict.h mcpredict.hh mcsolver.h mcsolver.hh boolmatrix.h EigenIO.h typedefs.h constants.h normalize.h evaluate.h evaluate.hh mutexlock.h filter.h find_w.h KMeans.h parameter.h parameter-args.h mcsolveProg.hpp mcprojProg.hpp predict.h predict.hh PredictionSet.h printing.h printing.hh utils.h WeightVector.h find_w_detail.h find_w_detail.hh find_w.hh LPSR.h 
# faster version (can run in parallel)
check-headers chk-headers:
	rm -f /tmp/*.chk
	$(MAKE) -f$(THIS_MAKEFILE) -j10 $(patsubst %,/tmp/%.chk,$(CHECK_HEADERS))
check-oct-headers: EigenOctave.h
	for header in $^; do \
		echo "Checking standalone compile of $$header..."; \
		$(CXX) -c $$header $(CPPFLAGS) $(OCTAVE_CFLAGS) -o /dev/null; \
	done; \
	echo "DONE checking octave-related standalone header compilation"

LIB_CPPFILES:= mcpredict.cpp mcsolver.cpp find_w.cpp find_w_detail.cpp filter.cpp PredictionSet.cpp printing.cpp utils.cpp EigenIO.cpp parameter.cpp parameter-args.cpp evaluate.cpp predict.cpp normalize.cpp mcsolveProg.cpp mcprojProg.cpp optimizelu.cpp
lib: hdr-stamp
	$(MAKE) -f$(THIS_MAKEFILE) -j15 libs-only demo-proj1 demo-quick || { echo -e '\n\n\n\n\n'; $(MAKE) -f$(THIS_MAKEFILE) libs-only demo-proj1 demo-quick; }
lib-clean:
	rm -f libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so *.o *.lo
hdr-stamp: erik/hdr-stamp
erik/hdr-stamp: $(CHECK_HEADERS)
	mkdir -p erik
	$(MAKE) -f$(THIS_MAKEFILE) lib-clean ####### && $(MAKE) -f$(THIS_MAKEFILE) -j15 lib || { reset; $(MAKE) -f$(THIS_MAKEFILE) lib; }
	touch $@
libs-only: | hdr-stamp
libs-only: libmcfilter.a libmcfilter-dbg.a libmcfilter.so libmcfilter-dbg.so
#demp-proj1 demo-quick demo-proj2 demo-proj3 demo-proj4 mcdumpsoln demo-prof: | libs-only

test: lib demo-printing demo-io demo-norm do-tests mcgen mcgenx mcproj.log
long: clean
	mkdir -p erik
	{ $(MAKE) -f$(THIS_MAKEFILE) -j20 demo-mcsolver && ./demo-mcsolver 2>&1 | tee dmc.log | (head -n5; echo "^^^... ... ...vvv"; tail -n20); } \
	&& $(MAKE) -f$(THIS_MAKEFILE) -j20 do-tests \
	&& $(MAKE) -f$(THIS_MAKEFILE) -j20 demo-proj1 demo-proj-dbg demo-proj2-dbg \
	&& { if [ -f dpa.log ]; then tail -n5 dpa.log; else $(MAKE) -f$(THIS_MAKEFILE) dpa.log; fi; } \
	&& $(MAKE) -f$(THIS_MAKEFILE) dpb.log
	rm -f dpa1.log dpb1.log dpa2.log dpb2.log dpa3.log dpb3.log dpa4.log dpb4.log dpa5.log dpb5.log dpa6.log dpb6.log
	$(MAKE) -f$(THIS_MAKEFILE) -j6 dpa1.log dpb1.log dpa2.log dpb2.log dpa3.log dpb3.log dpa4.log dpb4.log dpa5.log dpb5.log dpa6.log dpb6.log dpa7.log dpb7.log dpa8.log dpb8.log > erik/mk-long.log 2>&1
	$(MAKE) -f$(THIS_MAKEFILE) long-compare
long-compare:
	{ for n in "" 1 2 3 4 5 6 7 8; do \
	  a=`tail dpa$$n.log | grep '^obj' | tail -n2 | head -n1`; \
	  b=`tail dpb$$n.log | grep '^obj' | tail -n2 | head -n1`; \
	  echo "dpa$$n: $$a     dpb$$n: $$b"; \
	done; }
	{ for n in "" 1 2 3 4 5 6 7 8; do \
	  a=`tail dpa$$n.log | grep '^obj' | tail -n1`; \
	  b=`tail dpb$$n.log | grep '^obj' | tail -n1`; \
	  echo "Dpa$$n: $$a     Dpb$$n: $$b"; \
	done; }
	echo 'OK: make long'
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
dpa7.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 7 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb7.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 7 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpa8.log: demo-proj-dbg | libmcfilter-dbg.a
	@echo -e "\n$@ ..."
	./$< 8 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
dpb8.log: demo-proj2-dbg | libmcfilter-dbg.a dpa.log
	@echo -e "\n$@ ..."
	./$< 8 2>&1 | tee $@ | (head -n15; echo "^^^... tail $@ ...vvv"; tail -n5; echo "----    ----    ----"; )
do-vg: demo-mcsolver
	mkdir -p erik
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver m 2>&1 | tee dmc-m.txt | tail
	@echo ""
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver o 2>&1 | tee dmc-o.txt | tail
	@echo -n "valgrind tests 0-C ... "
	for t in 0 1 2 3 4 5 6 7 8 9 A B C; do  \
	  echo -n " $$t"; \
	  valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver op0$$t > erik/log-op0$$t.vg 2>&1 & \
	  valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver mp0$$t > erik/log-mp0$$t.vg 2>&1 & \
	  wait; \
	done;
	@echo DONE valgrind tests
	$(MAKE) -f$(THIS_MAKEFILE) tail-vg
tail-vg:
	tail -n1 erik/log-op0*.vg
	tail -n1 erik/log-mp0*.vg
do-tests: | lib demo-printing demo-io demo-norm demo-quick demo-mcsolver test-scaling.log
do-tests: | lib demo-printing demo-io demo-norm demo-quick demo-mcsolver test-scaling.log
	./demo-quick > demo-quick.log 2>&1
	./demo-norm > demo-norm.log 2>&1
	./demo-printing > demo-printing.log 2>&1
	./demo-io > demo-io.log 2>&1
	./demo-mcsolver > demo-mcsolver.log 2>&1
	 valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver m 2>&1 | tee dmc-m.txt | tail
	 valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -v ./demo-mcsolver o 2>&1 | tee dmc-o.txt | tail
	# All the simple tests (p0) can converge, for both original (o) and mcsolver (m,default)
	@nfails=0; ntests=0; \
	for t in 0 1 2 3 4 5 6 7 8 9 A B C; do  \
	  { ./demo-mcsolver op0$$t > erik/log-op0$$t.log 2>&1 || { nfails=$$(($$nfails+1)); echo "ERROR in demo-mcsolver op0$$t"; false; } ; } &\
	  { ./demo-mcsolver mp0$$t > erik/log-mp0$$t.log 2>&1 || { nfails=$$(($$nfails+1)); echo "ERROR in demo-mcsolver mp0$$t"; false; } ; } &\
	  wait; \
	  ntests=$$(($$ntests+1)); \
	done; \
	if [ "$$nfails" = "0" ]; then echo "                do-tests *** PASSED $$ntests TESTS ***"; \
	else                          echo "                do-tests *** FAILED $$nfails TESTS ***"; false; fi
	@echo ""
	@echo ""

# Create a library of octave-free C++ code for the main functionality in MCFilter
#   (find_w, filter, kmeans, LPSR)
#   This should aid in producing an interface in lua [I hope]
# XXX predict.cpp and evaluate.cpp MIGHT be removable?
libmcfilter.a: $(patsubst %.cpp,%.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter-dbg.a: $(patsubst %.cpp,%-dbg.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter.so: $(patsubst %.cpp,%.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^ $(BOOSTLINK) -lboost_program_options
libmcfilter-dbg.so: $(patsubst %.cpp,%-dbg.lo,$(LIB_CPPFILES))
	$(CXX) -o $@ -shared $^ $(BOOSTLINK) -lboost_program_options
# special library versions
libmcfilter-prof.a: $(patsubst %.cpp,%-prof.o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter2.a: $(patsubst %.cpp,%.2o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter3.a: $(patsubst %.cpp,%.3o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^
libmcfilter4.a: $(patsubst %.cpp,%.4o,$(LIB_CPPFILES))
	$(AR) -vru $@ $^

# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
%.2o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -DMCTHREADS=0 -c $<
%.3o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -D_GLIBCXX_PARALLEL -c $<
%.4o: %.cpp
	$(CXX) -o $@ $(CPPFLAGS) -DMCTHREADS=0 -D_GLIBCXX_PARALLEL -c $<
%-prof.o: %.cpp
	$(CXX) -o $@ $(PROFILE_CPPFLAGS) -c $<
demo-proj1: | libmcfilter.a
demo-proj1: erik/demo-proj.cpp libmcfilter.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) -I$(MAKEFILE_DIR)/..  $(CPPFLAGS) $^ $(LDFLAGS)
	@#/usr/bin/time -v ./$@ 2>&1 | tee $@.log
demo-proj2: erik/demo-proj.cpp libmcfilter2.a
	$(CXX) -o $@ -DMCTHREADS=0 -I$(MAKEFILE_DIR) $(CPPFLAGS) $^ $(LDFLAGS)
demo-proj3: erik/demo-proj.cpp libmcfilter3.a
	$(CXX) -o $@ -D_GLIBCXX_PARALLEL -I$(MAKEFILE_DIR) $(CPPFLAGS) $^ $(LDFLAGS) -lgomp
demo-proj4: erik/demo-proj.cpp libmcfilter4.a
	$(CXX) -o $@ -DMCTHREADS=0 -D_GLIBCXX_PARALLEL -I$(MAKEFILE_DIR)  $(CPPFLAGS) $^ $(LDFLAGS) -lgomp
demo-prof: erik/demo-proj.cpp libmcfilter-prof.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(PROFILE_CPPFLAGS) $^ $(PROFILE_LDFLAGS)
demo-quad: erik/demo-quad.cpp
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(CPPFLAGS) $^
do-proj: demo-proj1 demo-proj2 demo-proj3 demo-proj4
	/usr/bin/time -v ./demo-proj1 2>&1 | tee demo-proj1.log
	/usr/bin/time -v ./demo-proj2 2>&1 | tee demo-proj2.log
	/usr/bin/time -v ./demo-proj3 2>&1 | tee demo-proj3.log
	OMP_NUM_THREADS=4 /usr/bin/time -v ./demo-proj3 2>&1 | tee demo-proj3-omp8.log # twice as slow
	/usr/bin/time -v ./demo-proj4 2>&1 | tee demo-proj4.log
	@echo "\t\t\t make do-proj COMPLETE"
	@echo "\t\t\t see demo-proj1.log demo-proj2.log demo-proj3.log demo-proj3-omp8.log demo-proj4.log"

# With AVX2 on snake06 (no openMP) the following test takes ~ 64 s
demo-proj-dbg: | libmcfilter-dbg.a
demo-proj-dbg: erik/demo-proj.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -DUSE_MCSOLVER=0 -I$(MAKEFILE_DIR) $(CPPFLAGS) $^ $(LDFLAGS)
demo-proj2-dbg: | libmcfilter-dbg.a
demo-proj2-dbg: erik/demo-proj.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -DUSE_MCSOLVER=1 -I$(MAKEFILE_DIR) $(CPPFLAGS) $^ $(LDFLAGS)
# The following test throws if the solution is not sufficiently close to the correct answer.
# Note: inside eigen, there are signed vs. unsigned issues (error=sign-compare)
# Warnings:
# 	eigen has internal 'sign-compare' issues
# 	std::random* has internal 'maybe-uninitialized' issues
demo-quick: | libmcfilter-dbg.a
demo-quick: erik/demo-quick.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^ $(LDFLAGS)
demo-mcsolver: | libmcfilter-dbg.a
demo-mcsolver: erik/demo-mcsolver.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^ $(LDFLAGS)
test-scaling.log: demo-mcsolver
	@echo "Test simple problem with xscale 1.0 and 2.0" > $@
	./demo-mcsolver -p0v >> $@
	@echo -e "\n\n\n Same problem scaling all x data and C1,C2 by 2.0" >> $@
	./demo-mcsolver -p0vx >> $@
	# 2 solutions in $@ should have {lb,ub} different by roughly a factor of 2.0
demo-printing: erik/demo-printing.cpp
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
demo-parms: | libmcfilter-dbg.a
demo-parms: erik/demo-parms.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^ $(LDFLAGS)

demo-io: | libmcfilter-dbg.a
demo-io: erik/demo-io.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
demo-norm: | libmcfilter-dbg.a
demo-norm: erik/demo-norm.cpp libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
#static linkage : 2.6M executable
#mcgen: erik/mcgen.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.
mcgen: erik/mcgen.cpp
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^ $(BOOSTLIB)/libboost_program_options.a
# link with libmcfilter allows saving a .soln file ...
mcgenx: | libmcfilter-dbg.a
mcgenx: erik/mcgen.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a
	$(CXX) -o $@ -DUSE_LIBMCFILTER -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
mcdumpsoln: erik/mcdumpsoln.cpp libmcfilter-dbg.a | libmcfilter-dbg.a
	$(CXX) -o $@ -I$(MAKEFILE_DIR) $(DBGFLAGS) -Wno-error=maybe-uninitialized $^
erik/mcsolve0: erik/mcsolve0.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a
	# mcsolve0 is an initial version, before opt::MCsolveProgram was written
	mkdir -p erik
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
mcsolve: mcsolve.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
mcproj0: erik/mcproj.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
mcproj: mcproj.cpp libmcfilter-dbg.a $(BOOSTLIB)/libboost_program_options.a
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(DBGFLAGS) $^
.PHONY: run-mcproj
run-mcproj:
	rm -f mcproj mcproj0; $(MAKE) -f$(THIS_MAKEFILE) hdr-stamp
	make -j20 mcproj mcproj0 mcgenx || { make mcproj mcproj0 mcgenx; }
	./mcgenx > mcgenx.log 2>&1
	./mcproj0 --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin > s0.log 2>&1 & \
	./mcproj --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin > s.log 2>&1 & \
	wait && echo "gvim s.log s0.log -O # <-- outputs"
mcproj.log: mcgenx mcproj mcsolve erik/mcsolve0
	./mcgenx
	{ for cmd in "valgrind --leak-check=full --show-leak-kinds=all ./mcproj --solnfile mcgen-a3-txt.soln -x mcgen-a3-x-D.bin" \
		"valgrind ./mcproj --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-D.bin" \
		"valgrind ./mcproj --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-S.bin" \
		"valgrind ./mcproj --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin" \
		"valgrind ./erik/mcsolve0 --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -o xx --reoptlu" \
		"valgrind ./mcsolve --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -o xx --reoptlu" \
		"valgrind ./mcproj --solnfile xx.soln -x mcgen-a3-x-D.bin" \
		; do \
		echo ""; \
		echo "=== run : $$cmd"; \
		$$cmd; \
		done; } 2>&1 | tee $@
	@echo ""
	@echo mcproj.log valgrind summary:
	@grep '\(ERROR SUMMARY\)\|\(== run :\)' mcproj.log
mcproj1.log: mcgenx mcproj mcsolve
	./mcgenx
	{ for cmd in \
		"./mcproj --solnfile=mcgen-a3-bin.soln -xmcgen-a3-x-D.bin -ymcgen-a3-slc-y.bin -v" \
		"./mcsolve --solnfile=mcgen-a3-bin.soln -xmcgen-a3-x-D.bin -ymcgen-a3-slc-y.bin -oxx --reoptlu --treport=500" \
		"./mcproj --solnfile=xx.soln -xmcgen-a3-x-D.bin -ymcgen-a3-slc-y.bin -v" \
		; do \
		echo ""; \
		echo "=== run : $$cmd"; \
		$$cmd; \
		done; } 2>&1 | tee $@
mcproj2.log: mcgenx mcproj mcsolve
	./mcgenx
	{ for cmd in \
		"./mcproj --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-mlc-y.bin" \
		"./mcsolve --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-mlc-y.bin -o xx --reoptlu" \
		"./mcproj --solnfile xx.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -v" \
		; do \
		echo ""; \
		echo "=== run : $$cmd"; \
		$$cmd; \
		done; } 2>&1 | tee $@
mcproj3.log: mcgenx mcproj mcsolve
	./mcgenx
	{ for cmd in "./mcproj --solnfile mcgen-a3-txt.soln -x mcgen-a3-x-D.bin" \
		"./mcproj --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-D.bin" \
		"./mcsolve --solnfile mcgen-a3-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -o xx --reoptlu --resume" \
		"./mcproj --solnfile xx.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -v" \
		; do \
		echo ""; \
		echo "=== run : $$cmd"; \
		$$cmd; \
		done; } 2>&1 | tee $@
mcproj4.log: mcgenx mcproj mcsolve
	./mcgenx
	{ for cmd in \
		"./mcproj --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-mlc-y.bin" \
		"./mcsolve --solnfile mcgen-a3-mlc-bin.soln -x mcgen-a3-x-D.bin -y mcgen-a3-mlc-y.bin -o xx --reoptlu --resume" \
		"./mcproj --solnfile xx.soln -x mcgen-a3-x-D.bin -y mcgen-a3-slc-y.bin -v" \
		; do \
		echo ""; \
		echo "=== run : $$cmd"; \
		$$cmd; \
		done; } 2>&1 | tee $@
#shared linkage : 2.1M executable.. Not much savings.
#mcgen: erik/mcgen.cpp | libmcfilter-dbg.so
#	$(CXX) -o $@ -fPIC -I$(MAKEFILE_DIR) $(DBGFLAGS) $< -Wl,-Bdynamic -L. -lmcfilter-dbg -L$(BOOSTLIB) -lboost_program_options -Wl,-rpath=$(BOOSTLIB) -Wl,-rpath=.

#mcsolve-opt-prof: mcsolve.cpp libmcfilter-prof.a $(BOOSTLIB)/libboost_program_options.a -lprofiler /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
#mcsolve-opt-prof: mcsolve.cpp libmcfilter-prof.a $(BOOSTLIB)/libboost_program_options.a -lprofiler /usr/lib/debug/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 -lpthread /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 
mcsolve-opt-prof: mcsolve.cpp libmcfilter-prof.a $(BOOSTLIB)/libboost_program_options.a -lprofiler
	$(CXX) -o $@ -Wno-error=maybe-uninitialized -I$(MAKEFILE_DIR) $(CPPFLAGS) -DPROFILE $^

#TEST_ARGS:=--xfile=../../data/temp.test.svm --output=big.soln -B -S --C1=-1 --maxiter=100 --optlu=10 --treport=10 --proj=1 --threads=0
TEST_ARGS:=--xfile=../../data/temp.test.svm --output=big.soln -B -S --C1=-1 --maxiter=1000000 --optlu=10000 --treport=50000 --proj=1 --threads=0
#TEST_ARGS:=--xfile=../../data/temp.test.svm --output=big.soln -B -S --C1=-1 --maxiter=1000 --optlu=1 --treport=100 --proj=1 --threads=0	 # bang on --optlu

LEARNING_PROFILE:=learning.profile
# If you install the libgomp1-dbg package, then you can get nicer kcachegrind summaries...
LEARNING_PROFILE+=--lib_prefix=/usr/lib/debug/usr/lib/x86_64-linux-gnu --add_lib=libgomp.so.1.0.0
# currently v8.log has the shortest wall time. (24.13 s, cf 55 s before working on omp stuff)
# For 4 threads, u4.log has shortest wall time (28.92 s, but v4 was not far behind: 29.17 s)
t%.log:
	export OMP_DISPLAY_ENV=VERBOSE; \
		export OMP_NUM_THREADS=$*; export OMP_THREAD_LIMIT=16; export OMP_DYNAMIC=FALSE; \
		{ rm -f *.profile mcsolve mcsolve-opt-prof; \
		$(MAKE) -f$(THIS_MAKEFILE) hdr-stamp; $(MAKE) -f$(THIS_MAKEFILE) -j6 mcsolve mcsolve-opt-prof libmcfilter-prof.a; \
		/usr/bin/time -v ./mcsolve-opt-prof $(TEST_ARGS); } 2>&1 | tee $@; \
		google-pprof --callgrind ./mcsolve-opt-prof $(LEARNING_PROFILE) > t$*.callgrind 2> t$*.err; \
		echo ......... kcachegrind t$*.callgrind
u%.log:
	export OMP_DISPLAY_ENV=VERBOSE; \
		export OMP_NUM_THREADS=$*; export OMP_DYNAMIC=FALSE; export OMP_NESTED=TRUE; export OMP_THREAD_LIMIT=8; \
		{ rm -f *.profile mcsolve mcsolve-opt-prof; \
		$(MAKE) -f$(THIS_MAKEFILE) hdr-stamp; $(MAKE) -f$(THIS_MAKEFILE) -j6 mcsolve mcsolve-opt-prof libmcfilter-prof.a; \
		/usr/bin/time -v ./mcsolve-opt-prof $(TEST_ARGS) --update=SAFE; } 2>&1 | tee $@; \
		google-pprof --callgrind ./mcsolve-opt-prof $(LEARNING_PROFILE) > u$*.callgrind 2> u$*.err; \
		echo ......... kcachegrind u$*.callgrind
v%.log:
	export OMP_DISPLAY_ENV=VERBOSE; \
		export OMP_NUM_THREADS=$*; export OMP_DYNAMIC=FALSE; export OMP_NESTED=FALSE; export OMP_THREAD_LIMIT=$*; \
		{ rm -f *.profile mcsolve mcsolve-opt-prof; \
		$(MAKE) -f$(THIS_MAKEFILE) hdr-stamp; $(MAKE) -f$(THIS_MAKEFILE) -j6 mcsolve mcsolve-opt-prof libmcfilter-prof.a; \
		/usr/bin/time -v ./mcsolve-opt-prof $(TEST_ARGS) --update=SAFE; } 2>&1 | tee $@; \
		google-pprof --callgrind ./mcsolve-opt-prof $(LEARNING_PROFILE) > v$*.callgrind 2> v$*.err; \
		echo ......... kcachegrind v$*.callgrind
w%.log:
	export OMP_DISPLAY_ENV=VERBOSE; \
		export OMP_NUM_THREADS=$*; export OMP_DYNAMIC=FALSE; export OMP_NESTED=TRUE; export OMP_THREAD_LIMIT=$$(($**2)); \
		{ rm -f *.profile mcsolve mcsolve-opt-prof; \
		$(MAKE) -f$(THIS_MAKEFILE) hdr-stamp; $(MAKE) -f$(THIS_MAKEFILE) -j6 mcsolve mcsolve-opt-prof libmcfilter-prof.a; \
		/usr/bin/time -v ./mcsolve-opt-prof $(TEST_ARGS) --update=SAFE; } 2>&1 | tee $@; \
		google-pprof --callgrind ./mcsolve-opt-prof $(LEARNING_PROFILE) > w$*.callgrind 2> w$*.err; \
		echo ......... kcachegrind w$*.callgrind
x%.log:
	export OMP_DISPLAY_ENV=VERBOSE; \
		export OMP_NUM_THREADS=$*; export OMP_DYNAMIC=FALSE; export OMP_NESTED=TRUE; export OMP_THREAD_LIMIT=$$(($**4)); \
		{ rm -f *.profile mcsolve mcsolve-opt-prof; \
		$(MAKE) -f$(THIS_MAKEFILE) hdr-stamp; $(MAKE) -f$(THIS_MAKEFILE) -j6 mcsolve mcsolve-opt-prof libmcfilter-prof.a; \
		/usr/bin/time -v ./mcsolve-opt-prof $(TEST_ARGS) --update=SAFE; } 2>&1 | tee $@; \
		google-pprof --callgrind ./mcsolve-opt-prof $(LEARNING_PROFILE) > x$*.callgrind 2> x$*.err; \
		echo ......... kcachegrind x$*.callgrind

clean: 
	rm -f *.oct *.o *.lo *-dbg.o *-dbg.lo *.2o *.3o *.4o *-prof.o libmcfilter*.a libmcfilter*.so \
		learn_projection evaluate_projection run_kmeans run_LPSR naive_bayes \
		mcsolve mcdumpsoln mcgenx erik/mcsolve0 erik/hdr-stamp \
		demo-quick demo-proj1 demo-proj-dbg \
		demo-quick demo-printing demo-io demo-quick.log demo-printing.log demo-quick.log \
	       	demo-proj2-dbg demo-mcsolver demo-proj.native mcgen demo-proj[1-4] demo-prof xx a.out \
		demo-parms

realclean: clean
	rm -f *.log *.txt DenseM.bin *.soln *.repo boolmatrix.bin *.profile mcgen*.bin
