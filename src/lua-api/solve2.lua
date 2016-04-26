dofile('repo2xy-slc.lua')
print(" Created xxx.bin and yyy.txt for slc demo of MCFilter solver")
-- now xxx.bin and yyy.txt are binary dense data and single-class text
p=require('mcparm')
print(libmclua.mcsolve.help())
-- -B -S for Binary Short --output
-- -T -L for Text   Long  --output
solveArgs="--xfile=xxx.bin --yfile=yyy.txt --output=solve2-BS.soln -B -S --maxiter=5000"
s=libmclua.mcsolve.new(solveArgs)
s:read():solve():save():display()
print(" Solved xxx.bin and yyy.txt for slc demo")
print(" Used s=libmclua.mcsolve.new(\""..solveArgs.."\")")
print(" And then s:read():solve():save():display()")
print(" mcdumpsoln < solve2-BS.soln    to dump .soln file as text")
