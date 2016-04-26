require('milde')
require('mcparm')
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
solveArgs="--xfile=../../data/mnist.svm --output=mnist-quad.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=5000 --proj=10"
--
-- safe update requires x.row(i) norms to all be one
--solveArgs="--xfile=../../data/mnist.svm --output=mnist-quad.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=25000 --update=SAFE --batchsize=1 --xnorm --proj=10"
s=libmclua.mcsolve.new(solveArgs)
s:read()
s:savex("mnist-x.bin")
s:savey("mnist-y.bin")
-- mnist-y.bin 176k --- text format might even be better ??

milde_core.shell.omp_threads( milde_core.shell.omp_procs() )
print(" omp_threads = omp_procs = "..milde_core.shell.omp_threads())
-- the next operation "converts" x data to add all quadratic dimensions
-- This runs in 17 Gb memory, as opposed to doing it in lua (80 Gb)
--
-- The savex stores a binary sparse version in 11 Gb "mnist-xquad.bin"
-- as opposed to milde's libsvm test formt of 139 Gb "quad-mnist.svm"
-- (and quad + savex finishes in just a few minutes, not hours)

t0 = milde_core.shell.time()
s:quadx()

t1 = milde_core.shell.time()
print(" s:quad() conversion took "..(t1-t0).." seconds")
-- Aaaah. much better.  18 s on a 16-processor box.

--s:savex("mnist-xquad.bin")
--t2 = milde_core.shell.time()
--print(" s:savex(<fname:str>) took "..(t2-t1).." seconds")
--  snake08 save took 53 s

-- until we figure out how to really SPEED things up ...
milde_core.shell.omp_threads( 1 )
s:solve():save():display()
print("\n Solved mnist.svm with quadratic dimensions")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():quadx():solve():save():display()")
print(" mcdumpsoln < mnist-quad.soln    to dump .soln file as text")
