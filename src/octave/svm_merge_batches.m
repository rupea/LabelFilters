function [out_final] = svm_merge_batches(filename, label_range, noClasses, cur_file)

start = 1;    
for lbl_idx = 1 : length(label_range) - 1
    load( cur_file(lbl_idx) );
    if ~exist('out_final', 'var'),
        out_final = zeros(size(out,1),noClasses);    
    end            
    out_final(:,start:start+size(out,2)-1)=out;
    start = size(out,2)+start;
end  
       
save(filename, 'out_final');    
  
    
return ;