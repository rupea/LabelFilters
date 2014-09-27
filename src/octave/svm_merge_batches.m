function [out_final, out_final_tr] = svm_merge_batches(filename, label_range, noClasses, cur_file)

start = 1;    
for lbl_idx = 1 : length(label_range) - 1
    load( cur_file(lbl_idx) );
    if ~exist('out_final', 'var'),
        out_final = zeros(size(out,1),noClasses);
	if (nargout == 2)
	  out_final_tr = zeros(size(out_tr,1),noClasses);
	endif
    end            
    out_final(:,start:start+size(out,2)-1)=out;
    if (nargout == 2)
      out_final_tr(:,start:start+size(out,2)-1)=out_tr;
    endif
    start = size(out,2)+start;
end  
       
save(filename, 'out_final', 'out_final_tr');    
  
    
return ;
