require 'milde'
-- || loaded module 'milde_gui'  
mc=require 'mcparm' -- this is a link to either libmclua.so or libmclua-dbg.so
print(mc:type())    -- mcparm
x=mc                --x=mc.new() or x=libmclua.mc.new()  -- NO LONGER NEEDED
-- scr_MCparm::new_stack! > print(x:type())
print(x:str())      -- only the non-default values, one per line
print(x:str(true))  -- true for "verbose" list all values, even if same as default
print(x:pretty())   -- nicer format for print(x:str(true))
print(x.help())     -- help for x:set({cmd_line_args:str})
--             no_projections 5
t=x:get()           -- all are at default value, so empty ArgMap table
for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
--
t=x:get(true)       -- true means 'all' entries (even if value == default)
for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
-- C1	10 -- etc.  (no particular order of output) -- ERROR ---------------- t["no_projections"] = 88
t.no_projections=77    -- a non-default value
x:set(t)              -- (actually returns a lua table of the new values)
t2=x:get()            -- get table with the one non-default setting
print("\nt2=x:get(), with update no_projections...")
for k,v in pairs(t2) do print(string.format("%30s",k),type(v),v) end
assert( t2.no_projections == 77 )
t3=x:getargs()            -- get table with the one non-default setting
print("\nt3=x:getargs(), "..type(t3)..", with update no_projections...")
for k,v in pairs(t3) do print(string.format("%30s",k),type(v),v) end
assert( tonumber(t3.no_projections) == 77 )

x=libmclua.mc.new()
print("\ntest x:setargs(t) supplying a string key, x:set(t) for t = ...")
t={}
t.no_projections="88.3" -- Args supposedly want string values
-- t.no_projections=88.4   -- but this is also accepted
t.eta_type = "ETA_CONST"
t.eta_type = "eta_const"   -- also acceptable
t.eta_type = "const"       -- a substring may also be acceptable (if you choose the right substring)
--t.eta_type = "garbage"     -- ---> ERROR
for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
x:setargs(t)
-- print("\tAny changes in Args t after setargs? (case conversion)")
-- for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
print("\tnow x:get()...")
t2=x:get();
for k,v in pairs(t2) do print(string.format("%30s",k),type(v),v) end
assert( t2.no_projections == 88 );
assert( t2.eta_type == "ETA_CONST" );
-- now same for x:set(t) ...
x=libmclua.mc.new()
print("\ntest x:set(t) supplying a string key, x:set(t) for t = ...")
t={}
t.no_projections=88        -- ArgMap supposedly wants int values
t.no_projections=88.3      -- ok, but the .3 is ignored --t.no_projections="88.3"  -- --> ERROR
t.eta_type = "ETA_CONST"
t.eta_type = "eta_const"   -- also acceptable --t.eta_type = "garbage"     -- ---> ERROR
t.update_type = "SAFE_SGD"
t.update_type = "safe"     -- a substring may also be acceptable (if you choose the right substring)
for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
x:set(t)
print("\ntest x:set(\"--opt1=foo --opt2=bar ...\") supplying plain cmdline args as per print(x:help())")
x:set("--eta0=0.555 --etamin=0.222");
-- print("\tAny changes in Args t after setargs? (case conversion)")
-- for k,v in pairs(t) do print(string.format("%30s",k),type(v),v) end
print("\tnow x:get()...")
t2=x:get();
for k,v in pairs(t2) do print(string.format("%30s",k),type(v),v) end
assert( t2.no_projections == 88 );
assert( t2.eta_type == "ETA_CONST" );
assert( t2.update_type == "SAFE_SGD" );

p=libmclua.mc.new()
t={}
t.no_projections = 7.7777        -- int, so .77777 ignored
t.avg_epoch = 10                 -- size_t
t.update_type = "MINIBATCH_SGD"  -- enum, using full value
t.eta_type = "sqrt"              -- enum, using a lowercase substr
t.C1 = 7.7777                    -- double, so .7777 not ignored
t.resume = true                     -- bool (1 will not work for 'set')
p:set(t)                      -- modify p, MCfilter parameters
full=p:get(true)      -- get (true => ALL) things and verify
for k,v in pairs(full) do print(string.format("%30s",k),type(v),v) end
assert( full.no_projections == 7 );
assert( full.avg_epoch      == 10 );
assert( full.update_type    == "MINIBATCH_SGD" );
assert( full.eta_type       == "ETA_SQRT" );
assert( full.C1             == 7.7777 );
assert( full.resume         == true );
print("x:get(true)[\"resume\"]) = ", x:get(true)["resume"])  -- COULD overload x[string] ???
print("\nGOOD -- basic MCFilter parms tests passed")