#!/bin/bash
#
# check out -h function
./mcgenx -h
./mcsolve -h
./mcproj -h
#
# Create 8 classes (3 axes binary split on each axis) embedded in 11 of 11 dimensions
#    single-label problem can be split perfectly by 3 11-dimensional projection lines
#./mcgenx -a3 -e11 -d11 -x2000 >& 11-gen.log
#     same but added 2000 more random data points
./mcgenx -a3 -e11 -d11 -x2000 >& 11-gen.log
tail -n4 11-gen.log
ls -l mcgen-a3e11d11*
#
# project suggested solutions on sparse/dense training data
#    slc has 1 class per partition, mlc has 2 classes per partition (not separable)
./mcproj --solnfile mcgen-a3e11d11-txt.soln     --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc.log
./mcproj --solnfile mcgen-a3e11d11-mlc-txt.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-mlc.log
#
# Solve with default args : ask for 5 lines splitting 11-D slc data,
#                           knowing that 3 lines can perfectly split the data
#./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-a >& 11-slc-a.log
#tail -n60 11-slc-a.log
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-a 2>&1 | tee 11-slc-a.log
#
# Project previous solution (it is quite poor!)
./mcproj --solnfile 11-slc-a.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-a.log

# Bad soln: all intervals zero-length
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-b \
	--C1 1 --C2 1 --treport 50000 \
	2>&1 | tee 11-slc-b.log \
	&& ./mcproj --solnfile 11-slc-b.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-b.log
#
# Need quite some fiddling to get a decent approximation to good solution
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-c \
	--treport 50000 --C1 64 --C2 8 --eta0 1.e-2 --etamin 1e-5 \
	2>&1 | tee 11-slc-c.log \
	&& ./mcproj --solnfile 11-slc-c.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-c.log
# less iterations, doing much better... except many false negatives (filtered out the correct class)
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-c \
	--treport 10000 --maxiter 50000 --C1 64 --C2 8 --eta0 1 --etamin 1 \
	2>&1 | tee 11-slc-c.log \
	&& ./mcproj --solnfile 11-slc-c.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-c.log
#
# Good, but some have NO available classes
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-d \
	--treport 20000 --maxiter 200000 --C1 64 --C2 8 --eta0 1.e-1 --etamin 1e-1 --optlu 5000 --avg 50000 -b 1000 \
	2>&1 | tee 11-slc-d.log \
	&& ./mcproj --solnfile 11-slc-d.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-d.log
#
# 1 or 2 classes allowed but absolutely crazy predictions
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-d \
	--treport 20000 --maxiter 200000 --C1 5 --C2 1 --eta0 1.e-1 --etamin 1e-2 --optlu 5000 --avg 10000 -b 1000 \
	2>&1 | tee 11-slc-d.log \
	&& ./mcproj --solnfile 11-slc-d.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-d.log
#
#
./mcsolve --xfile mcgen-a3e11d11-x-D.bin --yfile mcgen-a3e11d11-slc-y.bin --output 11-slc-d \
	--treport 20000 --maxiter 200000 --C1 64 --C2 10 --eta0 1.e-1 --etamin 1e-1 --optlu 5000 --avg 50000 -b 1000 \
	2>&1 | tee 11-slc-d.log \
	&& ./mcproj --solnfile 11-slc-d.soln --xfile mcgen-a3e11d11-x-D.bin 2>&1 | tee 11-proj-slc-d.log
#
