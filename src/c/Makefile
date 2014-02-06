CXX ?= g++
CPPFLAGS ?= -O3
LDFLAGS ?= ""
PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install
PROFILE_CPPFLAGS = -g -O2 -DPROFILE -I$(PROFILEDIR)/include 
PROFILE_LDFLAGS = -L$(PROFILEDIR)/lib -lprofiler

all: find_w

profile: 
	@make CPPFLAGS="$(PROFILE_CPPFLAGS)" LDFLAGS="$(PROFILE_LDFLAGS)" find_w

find_w: find_w.o printing.o 
	$(CXX) $(LDFLAGS) -o find_w find_w.o printing.o

.cpp.o:
	$(CXX) $(CPPFLAGS) -c $<

clean: 
	rm -f *.o find_w find_w.profile
