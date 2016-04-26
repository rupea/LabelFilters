require('milde')
require('mcparm')
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
solveArgs="--xfile=../../data/mnist.svm --output=mnist.soln -B -S --C1=10 --C2=1 --maxiter=20000000 --optlu=5000 --treport=100000 --proj=10"
s=libmclua.mcsolve.new(solveArgs)
s:read():solve():save():display()
print("\n Solved mnist.svm for 9 classes demo")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():solve():save():display()")
print(" mcdumpsoln < big.soln    to dump .soln file as text")
