require 'milde'
-- || loaded module 'milde_gui'  
p=require 'mcparm' -- this is a link to either libmclua.so or libmclua-dbg.so
print(p:type())    -- mcparm
-- hmmm. should have a separate mcsolver.cpp file to get the names nicely done
--s=libmclua.mcsolve.s_new("--maxiter=5000")  -- exception (require parameters missing)
s=libmclua.mcsolve.new("--xfile=foo.x --yfile=foo.y --solnfile=foo.soln --maxiter=5000")
print(s:type())   -- mcsolve
