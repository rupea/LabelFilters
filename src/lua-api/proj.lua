require 'milde'
parm=require 'mcparm'     -- but we do not use parm at all for projection
print(libmclua.mcproj.help())
os.execute("cd ../c && make mcgenx")
os.execute("../c/mcgenx")   -- generate some x,y,soln data files
proj=libmclua.mcproj.new("--xfile=mcgen-a3-x-D.bin --solnfile=mcgen-a3-bin.soln")
-- above has no yfile, no validatiion (not implemented)
-- above has no --output, so classes will just go to cout
proj:read():proj():save():validate()
-- now change to sparse --xfile, mlc soln, sparse output (to cout)
proj=libmclua.mcproj.new("--xfile=mcgen-a3-x-S.bin --solnfile=mcgen-a3-mlc-bin.soln -S")
proj:read():proj():save():validate()
