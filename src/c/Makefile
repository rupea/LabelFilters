PROFILEDIR = /home/mlshack/alex/Programs/gperftools-2.1/install

all:

profile: find_w.cpp
	g++ -g -O2 -I$(PROFILEDIR)/include -L$(PROFILEDIR)/lib -DPROFILE -lprofiler -o find_w find_w.cpp
