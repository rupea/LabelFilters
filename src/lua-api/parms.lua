require 'milde'
-- || loaded module 'milde_gui'  
mc=require 'mcparm'
x=mc.new()
-- scr_MCparm::new_stack! > print(x:type())
-- mcparm 
print(x:str())
print(x:str(true))  -- true for "verbose" list all values, even if same as default
--             no_projections 5
t=x:get()           -- all are at default value, so empty ArgMap table
for k,v in pairs(t) do print(k,v) end
--
t=x:get(true)       -- true means 'all' entries (even if value == default)
for k,v in pairs(t) do print(k,v) end
-- C1	10
-- etc.  (no particular order of output)
t.no_projections=77    -- a non-default value
-- ERROR ---------------- t["no_projections"] = 88
x:set(t)
t2=x:get()            -- get table with the one non-default setting
print("t2, with update no_projections...")
for k,v in pairs(t2) do print(k,v) end
-- ??? how to do this--- assert( t2.no_projections == 77 );
