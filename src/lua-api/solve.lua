require 'milde'
-- || loaded module 'milde_gui'  
p=require 'mcparm' -- this is a link to either libmclua.so or libmclua-dbg.so
print(p:type())    -- mcparm
print(libmclua.mcsolve.help())
-- hmmm. should have a separate mcsolver.cpp file to get the names nicely done
--s=libmclua.mcsolve.s_new("--maxiter=5000")  -- exception (require parameters missing)
os.execute("cd ../c && make mcgenx")
os.execute("../c/mcgenx")   -- generate some x,y,soln data files
s=libmclua.mcsolve.new("--xfile=mcgen-a3-x-D.bin --yfile=mcgen-a3-slc-y.bin --solnfile=mcgen-a3-bin.soln --maxiter=5000")
print(s:type())   -- mcsolve
print(s:help())

-- execute the <mcsolve> fully
s:read():solve():save():display()
-- any erroneous step will just throw, hopefully :)
-- Should each step return an error code instead of the lua <mcsolve> object?
