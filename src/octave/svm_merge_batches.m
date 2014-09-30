function [out_final, out_final_tr] = svm_merge_batches(filename, label_range, noClasses, cur_file)

  start = 1;    
  for lbl_idx = 1 : (length(label_range) - 1)
    start_range = label_range(label_idx);
    end_range = label_range(lbl_idx+1)-1;
    load( cur_file(lbl_idx) );
    if ~exist('out_final', 'var')
      out_final = zeros(size(out,1),noClasses);
      if (nargout == 2)
	out_final_tr = zeros(size(out_tr,1),noClasses);
      end
    end            
    out_final(:,start_range:end_range)=out;
    if (nargout == 2)
      out_final_tr(:,start_range:end_range)=out_tr;
    endif
  end  

  if (nargout == 2)
    save(filename, 'out_final', 'out_final_tr');    
  else
    save(filename, 'out_final');        
  end
  return ;
end