require('milde')
require('mcparm')
--
-- export OMP_DISPLAY_ENV=VERBOSE in shell before starting up lua_cpp
--
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
-- Originally, required x row norms one but --xnorm norms over cols! ?
--solveArgs="--xfile=../../data/mnist.svm --output=mnist-quad.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=25000 --update=SAFE --batchsize=1 --xnorm --proj=10"
--solveArgs="--xfile=../../data/mnist.svm --output=mnist-quad.soln -B -S --C1=10 --C2=1 --maxiter=1000000 --optlu=5000 --treport=5000 --proj=10"
--solveArgs="--xfile=../../data/mnist.svm --output=mnist-qtest.soln -B -S --C1=10 --C2=1 --maxiter=6000 --optlu=500 --treport=1000 --proj=1"
solveArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --output=mnist-q2a.soln -B -S --C1=60000 --C2=1 --maxiter=10000000 --optlu=5000 --treport=100000 --proj=10 --update=SAFE -b1 --threads=2 --eta0=0.1"
s=libmclua.mcsolve.new(solveArgs)
s:read()
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
--  snake08 save took 53 s --> 11G binary data

-- until we figure out how to really SPEED things up ...
thr=2
milde_core.shell.omp_threads( thr )
print("\n omp_threads("..thr..")\n")
t0 = milde_core.shell.time()
s:solve()
t1 = milde_core.shell.time()
print("\n mnist-q2a solved in "..(t1-t0).." seconds\n\n\n")
s:save():display()

-- XXX TODO proj:quad() function needs to be added !
projArgs="--xfile=mnist-xquad.bin --yfile=y-mnist.bin --solnfile=mnist-q2a.soln --output=mnist-q2a.proj"
proj=libmclua.mcproj.new(projArgs)
proj:read()
t=shell.time()
proj:proj()
tproj=shell.time()-t
proj:save():validate()
print("\n long run projected in "..tproj.." seconds\n\n\n")


print("\n mnist-q2a solved in "..(t1-t0).." seconds\n\n\n")
print("\n Solved mnist.svm with quadratic dimensions")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():quadx():solve():save():display()")
print(" mcdumpsoln < mnist-quad.soln    to dump .soln file as text")
