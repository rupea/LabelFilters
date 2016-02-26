-- show how to convert repo--> eigen formats
require("milde")
os.execute("make -j2 mcgen mcgenx") -- mcgenx also write a soln file
os.execute("mcgenx")
os.execute("ls -l mcgen-a3-*")
--
fa="mcgen-a3-slc-dr4.repo"
ra=repo("slc","dr4")
aa={mode="ascii", adata=fa}
ra:load(aa)
print("file "..fa..", info_size "..ra:info_size()..", data_size "..ra:data_size()..", data_dim "..ra:data_dim())
if ra:info_type() == "slc" then
  io.write("\tslc_size  "); ra:slc_size():print()
  io.write("\tslc_begin "); ra:slc_begin():print()
  io.write("\tslc_end   "); ra:slc_end():print()
end
io.write("\tsvc_label_count    "); print(ra:svc_label_count())
io.write("\tsvc_label_value(0)-> s = "); s=ra:svc_label_value(0); print(type(s),s)
io.write("\tsvc_label_index(s)       "); print(ra:svc_label_index(s))
--io.write("\tsvc_label_index   "); ra:slc_end():print()
if ra:info_type() == "slc" then
  print(fa.." dump:")
  for i = 0, ra:info_size()-1 do
    io.write(string.format("%-3u : ",i)); io.write(ra:get_info(i).."  "); ra:row(i):print()
  end
elseif ra:info_type() == "mlc" then
  print(fa.." dump:")
  for i = 0, ra:info_size()-1 do
    io.write(string.format("%-3u : ",i))
    tab= ra:get_info(i)
    for i=1, #tab, 1 do
      io.write(string.format("%-3u ",tab[i]))
    end
    io.write(": ")
    ra:row(i):print()
  end
else
  print("OOPS: unknown repo format")
end
--
print("mcgen-a3-x-D.bin, NOT class-ordered:")
os.execute("hexdump mcgen-a3-x-D.bin")
-------------------------------
--
-- generic Eigen binary x file -- BINARY DENSE
--
-------------------------------
os.execute("rm -f xxx.bin")
xxx=io.open("xxx.bin","wb")
u8(ra:data_size()):bsave(xxx)
u8(ra:data_dim()) :bsave(xxx)
for r=0, ra:data_size()-1, 1 do
  rr=ra:row(r)
  print(rr, rr:size(), r4(rr[0]), r4(rr[1]), r4(rr[2]))
  for c=0, ra:data_dim()-1, 1 do
    r4(rr[c]):bsave(xxx)
  end
end
xxx:close();
--------------------------------
--------------------------------
-- generic Eigen binary y file -- not really possible
--
-- generic Eigen TEXT y file is doable (matching C++ eigen_io_txtbool routine)
--
--------------------------------
os.execute("rm -f yyy.bin")
yyy=io.open("yyy.txt","w")
--u8(ra:info_size()):asave(yyy)
--u8(ra:data_dim()) :asave(yyy)
yyy:write(ra:info_size()); yyy:write("\n")
yyy:write(ra:svc_label_count()) ; yyy:write("\n")
ny=ra:info_size()
print(ra:info_type(), "ny=", ny)
s=ra:get_info(0)
print(type(s),s)
t=ra:get_info_index(0)
print(type(t),t)
if ra:info_type() == "slc" then
  print(" y SLC TEXT ... ")
  for i=0, ny-1, 1 do
    s = ra:get_info(i)  -- <str> for slc, {string} for mlc
    print(ra:svc_label_value(s))
    yyy:write( ra:svc_label_value( s )); yyy:write("\n")
  end
elseif ra:info_type() == "mlc" then
  print(" y MLC TEXT ... ")
  for i=0, ny-1, 1 do
    s = ra:get_info(i)  -- <str> for slc, {string} for mlc
    for i=1, #s, 1 do   -- {string} is a lua table, indexing starts at 1
      --print(ra:svc_label_value(s[i]))
      if( i > 1 ) then yyy:write(" ") end
      yyy:write( ra:svc_label_value(s[i]) )
    end
    yyy:write("\n")
  end
else
  print("ERROR: unsupported repo 'y' type")
end
yyy:close();
--------------------------------
--------------------------------
--
--   (can read the dense binary Eigen data into lua)
--
print("we wrote: (note that slc repo has class-ordered points")
os.execute("hexdump xxx.bin")
-- Reread dense, binary and print
xxr=io.open("xxx.bin","rb")
r=u8(); c=u8()
r:bload(xxr); print(r)
c:bload(xxr); print(c)
x=r4(); y=r4(); z=r4()
for i=0, r:lua()-1, 1 do
  -- here would really be a loop over 'c' cols, for dense read.
  x:bload(xxr); y:bload(xxr); z:bload(xxr)
  print(x,y,z)
end
xxr:close();
-- the files are NOT binary identical, but behave the same.
-- Why?
-- because repo ORDERS the slc repo by class!
--------------------------------
--
-- --> BINARY SPARSE is too difficult (use text format)
--
--------------------------------
print("TBD - 'x' data output in Eigen SPARSE TEXT format")
