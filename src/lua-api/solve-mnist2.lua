require('milde')
require('mcparm')
--
-- export OMP_DISPLAY_ENV=VERBOSE in shell before starting up lua_cpp
--
-- now xxx.bin and yyy.txt are binary dense data and single-class text
-- Note: mcparm links to either libmcfilter.so OR libmcfilter-dbg.so
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
if false then
  solveArgs="--xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=1000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  t1000=shell.time()-t
  s:save():display()
  print("\n optlu=1000 solved in "..t1000.." seconds\n\n\n")
  solveArgs="--xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  t5000=shell.time()-t
  s:save():display()
  print("\n optlu=5000 solved in "..t5000.." seconds\n\n\n")
  solveArgs="--xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=10000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  t10000=shell.time()-t
  s:save():display()
  print("\n optlu=10000 solved in "..t10000.." seconds\n\n\n")
  -- 1000: 7.8e7 --> 6.4e6 in 24.4 s
  -- 5000: 1.1e8 --> 5.54e6 in 10.6 s   <---- reasonable setting ?
  -- 10000: 9.6e7 --> 6.27e6 in 8.89 s
end
if false then
  solveArgs="--eta0=0.1 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n eta0=1e-1 solved in "..ta.." seconds\n\n\n")
  solveArgs="--eta0=1.e-2 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tb=shell.time()-t
  s:save():display()
  print("\n eta0=1e-2 solved in "..tb.." seconds\n\n\n")
  solveArgs="--eta0=1.e-3 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tc=shell.time()-t
  s:save():display()
  print("\n eta0=1e-3 solved in "..tc.." seconds\n\n\n")
  -- 1e-1  : 4.54e7 --> 4.06e6 in 7.45 s
  -- 1e-2  : 5.32e7 --> 5.95e6 in 7.43 s
  -- 1e-3  : 5.04e7 --> 6.77e6 in 7.47 s
  -- and NONE of these runs are monotonic.
  -- new parms 2.0 + 0.5 eta adjust, and use final eta in next calls too...
  -- 1e-1  : 1.30e8 --> 4.73e6 in 7.17 s
end
if false then
  solveArgs="--C1=10 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=10 solved in "..ta.." seconds\n\n\n")
  solveArgs="--C1=256 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tb=shell.time()-t
  s:save():display()
  print("\n --C1=256 solved in "..tb.." seconds\n\n\n")
  solveArgs="--C1=2500 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C2=1 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tc=shell.time()-t
  s:save():display()
  print("\n --C1=2500 solved in "..tc.." seconds\n\n\n")
  -- 10   : 6.56e7 --> 4.77e6   : 13.75x
  -- 256  : 1.25e9 --> 1.84e8   : 6.79x
  -- 2500 : 9.90e9 --> 1.45e9   : 6.82x
end
if false then
  solveArgs="--C1=10 --C2=1 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=10 --C2=1 solved in "..ta.." seconds\n\n\n")
  solveArgs="--C1=300 --C2=30 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tb=shell.time()-t
  s:save():display()
  print("\n --C1=300 --C2=30 solved in "..tb.." seconds\n\n\n")
  solveArgs="--C1=1000 --C2=100 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tc=shell.time()-t
  s:save():display()
  print("\n --C1=1000 --C2=100 solved in "..tc.." seconds\n\n\n")
  -- 10:1     : 6.51e7 --> 4.77e6   : 13.85x  widths 25-228
  -- 300:30   : 1.56e9 --> 2.24e8   : 6.96x   widths 1037-2418
  -- 1000:100 : 4.65e9 --> 5.89e8   : 7.89x   widths 5854-12973
end
if false then
  solveArgs="--C1=1 --C2=0.1 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=1 --C2=0.1 solved in "..ta.." seconds\n\n\n")
  solveArgs="--C1=0.1 --C2=0.01 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=0.1 --C2=0.01 solved in "..ta.." seconds\n\n\n")
  solveArgs="--C1=1 --C2=0.01 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=1 --C2=0.01 solved in "..ta.." seconds\n\n\n")
  -- 1:0.1    : 1.56e7 --> 982245  MONOTONICALLY widths from 5-36
  -- 0.1:0.01 : 7.60e6 --> 622339 monotonic
  -- 1:0.01   : 3.36e7 --> 1.31e6 monotonic
  solveArgs="--C1=10 --C2=0.1 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  ta=shell.time()-t
  s:save():display()
  print("\n --C1=10 --C2=0.1 solved in "..ta.." seconds\n\n\n") -- not monotonic, but widths 44-142
  -- changed code to NOT munge C1,C2 values
  -- a: wid 4-36    wnorm 0.103
  -- b: 2.5-10.6    wnorm 0.02
  -- c: 4.2-13.1    wnorm .04
  -- d: 59-198      wnorm .31
end
if false then
  solveArgs="--xfile=../../data/mnist.svm --output=mnist2c.soln -B -S --C1=1 --C2=0.1 --maxiter=10000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  s:savex("x-mnist.bin")
  s:savey("y-mnist.bin")
  print("\n test run solved in "..tlong.." seconds\n\n\n")
  -- svm format NOT accepted for projection, so use x-mnist.bin
  -- --yfile for projection validation is NOT well supported
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2c.soln --output=mnist2c.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n test run projected in "..tproj.." seconds\n\n\n")
  -- C1,C2    :  confusion matrix (tp,fp tn,fn)
  -- 0.1,0.01 :  54085, 175919  364081, 5915
  -- 1,0.1    :  51809, 163108  376892, 8191
  -- 10,1     :  48414, 142772  397228, 11586
  -- 1,10     :  3332, 874      539126, 56668
  -- 1,0.005  :  59981, 476746  63254, 19
  -- changed to maxiter 10e6 ---> not much change in conf. matrix
end
if false then  --------------------------------- long run ----------------------------------
   solveArgs="--xfile=../../data/mnist.svm --output=mnist2d.soln -B -S --C1=0.1 --C2=0.01 --maxiter=10000000 --optlu=5000 --treport=100000 --proj=30 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  s:savex("x-mnist.bin")
  s:savey("y-mnist.bin")
  print("\n long run solved in "..tlong.." seconds\n\n\n")
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2d.soln --output=mnist2d.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n long run projected in "..tproj.." seconds\n\n\n")
end
if false then  --------------------------------- long run ----------------------------------
   solveArgs="--xfile=../../data/mnist.svm --output=mnist2f.soln -B -S --C1=600 --C2=1 --maxiter=20000000 --optlu=5000 --treport=500000 --proj=30 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  --s:savex("x-mnist.bin")
  --s:savey("y-mnist.bin")
  print("\n long run solved in "..tlong.." seconds\n\n\n")
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2f.soln --output=mnist2f.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n long run projected in "..tproj.." seconds\n\n\n")
end
if false then  --------------------------------- long run ----------------------------------
  solveArgs="--xfile=../../data/mnist.svm --output=mnist2g.soln -B -S --C1=60 --C2=1 --maxiter=20000000 --optlu=5000 --treport=500000 --proj=30 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  thr=2
  milde_core.shell.omp_threads( thr )
  print("\n omp_threads("..thr..")\n")
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  --s:savex("x-mnist.bin")
  --s:savey("y-mnist.bin")
  print("\n long run solved in "..tlong.." seconds\n\n\n")
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2g.soln --output=mnist2g.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n long run projected in "..tproj.." seconds\n\n\n")
end
if false then  --------------------------------- long run ----------------------------------
  solveArgs="--xfile=../../data/mnist.svm --output=mnist2h.soln -B -S --C1=6 --C2=1 --maxiter=20000000 --optlu=5000 --treport=500000 --proj=30 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  thr=2
  milde_core.shell.omp_threads( thr )
  print("\n omp_threads("..thr..")\n")
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  --s:savex("x-mnist.bin")
  --s:savey("y-mnist.bin")
  print("\n long run solved in "..tlong.." seconds\n\n\n")
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2h.soln --output=mnist2h.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n long run projected in "..tproj.." seconds\n\n\n")
end
if true then  --------------------------------- long run ----------------------------------
  solveArgs="--xfile=../../data/mnist.svm --output=mnist2i.soln -B -S --C1=6 --C2=0.1 --maxiter=20000000 --optlu=5000 --treport=500000 --proj=30 --update=SAFE -b1 --threads=8 --eta0=0.1"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  thr=2
  milde_core.shell.omp_threads( thr )
  print("\n omp_threads("..thr..")\n")
  t=shell.time()
  s:solve()
  tlong=shell.time()-t
  s:save():display()
  --s:savex("x-mnist.bin")
  --s:savey("y-mnist.bin")
  print("\n long run solved in "..tlong.." seconds\n\n\n")
  projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2i.soln --output=mnist2i.proj"
  proj=libmclua.mcproj.new(projArgs)
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n long run projected in "..tproj.." seconds\n\n\n")
end

