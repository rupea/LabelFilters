require('milde')
require('mcparm')
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
--  NEW: if no --yfile, then try libsvm format, sparse x
solveArgs="--xfile=../../data/temp.test.svm --output=big.soln -B -S --maxiter=50000"
s=libmclua.mcsolve.new(solveArgs)
s:read():solve():save():display()
print(" Solved xxx.bin and yyy.txt for slc demo")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():solve():save():display()")
print(" mcdumpsoln < solve3-BS.soln    to dump .soln file as text")
