require 'milde'
-- || loaded module 'milde_gui'  
mc=require 'mcparm'
x=mc.new()
-- scr_MCparm::new_stack! > print(x:type())
-- mcparm 
print(x:str())
print(x:str(true))  -- true for "verbose" list all values, even if same as default
--             no_projections 5
-- ^D
