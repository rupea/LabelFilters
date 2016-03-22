-- Usage:
--   lua_cpp help-solve.lua
--
require('milde')
p=require('mcparm')
print(libmclua.mcsolve.help())
