require('milde')
require('mcparm')
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --maxiter=50000"
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --C2=0.01 --C1=1 --eta0=1. --etamin=1.e-5 --etatype=3_4 -b 1 --optlu=100 --maxiter=50000 --proj=1"
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --maxiter=2 --optlu=99999 --proj=1"
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --C2=0.01 --C1=1 --eta0=1. --etamin=1.e-5 --etatype=3_4 -b 1 --optlu=100 --maxiter=50000 --proj=1"
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --C1=20000 --C2=100 --maxiter=50000 --optlu=99999 --proj=1"
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --C1=20000 --C2=1 --maxiter=1000 --optlu=99999 --proj=1"
s=libmclua.mcsolve.new(solveArgs)
s:read():solve():save():display()
print("\n Solved xxx.bin and yyy.txt for slc demo")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():solve():save():display()")
print(" mcdumpsoln < big.soln    to dump .soln file as text")
