require "io"
require "milde"
require "os"
os.execute("ls -l")
fname="mnist.svm"
r=repo("slc","sr4")
r:load( {["mode"]="libsvm",["adata"]=fname} )
print(r)
r0=r:row(0).dup()
print(r0)
r0:print()
--
-- Milde bug?   d=r0:dpoint(768)  FAILS to convert spoint into dpoint
--
-- Explicitly expand sparse vector to dense outer product
d0=dpoint_r4(768*768,0)
for i = 0, 4 do
  if r0:val(i) > 0 then
    for j = 0, 4 do
      if r0:val(j) > 0 then
        print("d0["..r0:idx(i).."*768+"..r0:idx(j).."] = "..r0:val(i).." * "..r0:val(j))
        d0[r0:idx(i) * 768 + r0:idx(j)] = r0:val(i) * r0:val(j)
      end
    end
  end
end
s0=d0:spoint()
print(s0) s0:print()

d0=dpoint_r4(768*768,0)
for i = 0, r0:size()-1 do
  if r0:val(i) > 0 then
    for j = 0, r0:size()-1 do
      if r0:val(j) > 0 then
        d0[r0:idx(i) * 768 + r0:idx(j)] = r0:val(i) * r0:val(j)
      end
    end
  end
end
s0=d0:spoint()
print(s0); print(" s0.size()="..s0:size() )

print(r:get_info(0))

q=repo("slc","sr4")
q:push(r:get_info(0),s0)
q:save( {["mode"]="libsvm",["adata"]="quad-mnist.svm"} )

-- hmmm. the follow lua loop explodes to about 80G memory
--       and writes out a 139 Gb text file.
q=repo("slc","sr4")
for rr = 0, r:data_size()-1 do
  collectgarbage()
  sr=r:row(rr)                  -- sr = sparse row
  dq=dpoint_r4(768*768,0)       -- dq = dense quad ~ outer product
  for i = 0, sr:size()-1 do
    if sr:val(i) > 0 then
      for j = 0, sr:size()-1 do
        if sr:val(j) > 0 then
          d0[sr:idx(i) * 768 + sr:idx(j)] = sr:val(i) * sr:val(j)
        end
      end
    end
  end
  -- sq=d0:spoint()                -- sq = sparse quadratic
  q:push(r:get_info(rr), d0:spoint())
end
q:save( {["mode"]="libsvm",["adata"]="quad-mnist.svm"} )
--
-- so I'll try to provide a 'quad' method to move from original
-- to quadratic space
--
