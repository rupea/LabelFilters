require('milde')
require('mcparm')
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
solveArgs="--xfile=../../data/mnist-mlc.svm --output=mnist-mlc.soln -B -S --C1=10000 --C2=1 --maxiter=100000 --optlu=5000 --treport=50000 --proj=1"
--solveArgs="--xfile=../../data/mnist-mlc.svm --output=mnist.soln -B -S --C1=10 --C2=1 --maxiter=20000000 --optlu=5000 --treport=100000 --proj=10"
s=libmclua.mcsolve.new(solveArgs)
s:read()
t=shell.time()
s:solve()
tlong=shell.time()-t
s:save():display()
s:savex("x-mnist-mlc.bin")
s:savey("y-mnist-mlc.bin")
print("\n long run solved in "..tlong.." seconds\n\n\n")
projArgs="--xfile=x-mnist.bin --yfile=y-mnist.bin --solnfile=mnist2h.soln --output=mnist2h.proj"
proj=libmclua.mcproj.new(projArgs)
proj:read()
t=shell.time()
proj:proj()
tproj=shell.time()-t
proj:save():validate()
print("\n long run projected in "..tproj.." seconds\n\n\n")
print("\n Solved mnist-mlc.svm for 9 + ~60k classes demo")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():solve():save():display()")
print(" mcdumpsoln < big.soln    to dump .soln file as text")
