-- test program to read and print out some mcgen output
require("milde")
os.execute("make -j2 mcgen mcgenx") -- mcgenx also write a soln file
os.execute("mcgenx")
--
fa="mcgen-a3-slc-dr4.repo"
ra=repo("slc","dr4")
aa={mode="ascii", adata=fa}
ra:load(aa)
print("file "..fa..", info_size "..ra:info_size()..", data_size "..ra:data_size()..", data_dim "..ra:data_dim())
io.write("\tslc_size  "); ra:slc_size():print()
io.write("\tslc_begin "); ra:slc_begin():print()
io.write("\tslc_end   "); ra:slc_end():print()
print(fa.." dump:")
for i = 0, ra:info_size()-1 do
  io.write(string.format("%-3u : ",i)); io.write(ra:get_info(i).."  "); ra:row(i):print()
end
--
fb="mcgen-a3-mlc-dr4.repo"
rb=repo("mlc","dr4")
b={mode="ascii", adata=fb}
rb:load(b)
print("file "..fb..", info_size "..rb:info_size()..", data_size "..rb:data_size()..", data_dim "..rb:data_dim())
-- next 3 functions only apply to slc
--io.write("\tslc_size  "); rb:slc_size():print()
--io.write("\tslc_begin "); rb:slc_begin():print()
--io.write("\tslc_end   "); rb:slc_end():print()
print(fb.." dump:")
for i = 0, rb:info_size()-1 do
  io.write(string.format("%-3u : ",i))
  -- for mlc, 'y' values become a lua table
  for k,v in pairs(rb:get_info(i)) do io.write(string.format(" %7s",v)) end
  io.write("  "); rb:row(i):print()
end
--
fc="mcgen-a3-slc-sr4.repo"
rc=repo("slc","sr4")
c={mode="ascii", adata=fc}    -- mode="libsvm" also works (why is it NOT dealing with ':' as a separator?)
rc:load(c)
print("file "..fc..", info_size "..rc:info_size()..", data_size "..rc:data_size()..", data_dim "..rc:data_dim())
io.write("\tslc_size  "); rc:slc_size():print()
io.write("\tslc_begin "); rc:slc_begin():print()
io.write("\tslc_end   "); rc:slc_end():print()
print(fc.." dump:")
for i = 0, rc:info_size()-1 do
  io.write(string.format("%-3u : ",i))
  io.write(rc:get_info(i).."  ")
  rc:row(i):print()
end
--
fd="mcgen-a3-mlc-sr4.repo"
rd=repo("mlc","sr4")
ad={mode="ascii", adata=fd}
rd:load(ad)
print("file "..fd..", info_size "..rd:info_size()..", data_size "..rd:data_size()..", data_dim "..rd:data_dim())
print(fd.." dump:")
for i = 0, rd:info_size()-1 do
  io.write(string.format("%-3u : ",i))
  -- for mlc, 'y' values become a lua table
  for k,v in pairs(rd:get_info(i)) do io.write(string.format(" %7s",v)) end
  io.write("  "); rd:row(i):print()
end
--
fe="mcgen-a3-slc-dr4.repo"
re=repo("slc","dr4")
ae={mode="binary", adata=fe}
re:load(ae)
print("file "..fe..", info_size "..re:info_size()..", data_size "..re:data_size()..", data_dim "..re:data_dim())
io.write("\tslc_size  "); re:slc_size():print()
io.write("\tslc_begin "); re:slc_begin():print()
io.write("\tslc_end   "); re:slc_end():print()
print(fe.." dump:")
for i = 0, re:info_size()-1 do
  io.write(string.format("%-3u : ",i))
  io.write(re:get_info(i).."  ")
  re:row(i):print()
end
