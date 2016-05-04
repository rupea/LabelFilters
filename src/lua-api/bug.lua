require('milde')
require('mcparm')
p=require('mcparm')
if false then
  solveArgs="--C2=0.1 --xfile=../../data/mnist.svm --output=mnist-optlu1000.soln -B -S --C1=10 --maxiter=1000000 --optlu=5000 --treport=100000 --proj=1 --update=SAFE -b1 --threads=4"
  s=libmclua.mcsolve.new(solveArgs)
  s:read()
  t=shell.time()
  s:solve()
  tb=shell.time()-t
  s:save():display()
  print("\n --C2=0.1 solved in "..tb.." seconds\n\n\n")
end
if true then
  -- AHAA svm format input file is NOT SUPPORTED
  --   must use the solver:savex("x.bin") to get the dense binary x data.
  --   (ouch. no sparse save?)
  projArgs="--xfile=../../data/mnist.svm --solnfile=mnist2c.soln --output=mnist2c.proj"
  print(libmclua.mcproj.help())
  proj=libmclua.mcproj.new(projArgs)
  print(proj:help())
  proj:read()
  t=shell.time()
  proj:proj()
  tproj=shell.time()-t
  proj:save():validate()
  print("\n test run projected in "..tproj.." seconds\n\n\n")
end

