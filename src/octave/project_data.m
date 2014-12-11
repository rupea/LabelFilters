function [projected_labels, projected] = project_data(x, noLabels,w, min_proj, max_proj)
  
  %%project the instances and eliminate the filtered out classes
  projected = x * w;
  display(size(projected))
  projected_labels = cell(size(projected,2),1);
  for j = 1:size(projected,2)
    in_range_lbls = logical(sparse(size(projected,1),noLabels));
    display(j)
    for c = 1:noLabels
      in_range_lbls(:,c) = projected(:,j)>= min_proj(c,j) & projected(:,j) <= max_proj(c,j);
    end
    if (j>1)
      in_range_lbls = projected_labels{j-1} & in_range_lbls;
    end
    projected_labels{j}=in_range_lbls;
  end
  
  return;
end

# logical(ones(size(projected,1),noLabels,size(projected,2)));
#   for c = 1 : noLabels,
#     in_range_lbls = logical(ones(size(projected,1),1,size(projected,2)));
#     in_range_lbls(:,:,1) = projected(:,1) >= min_proj(c,1) &  projected(:,1) <= max_proj(c,1);
#     if ( size(projected,2) > 1 )
#       for j = 2 : size(projected,2),
#         in_range_lbls(:,:,j)  = and(in_range_lbls(:,:,j-1), projected(:,j) >= min_proj(c,j) &  projected(:,j) <= max_proj(c,j));
#       end
#     end
#     projected_labels(:,c,:) = in_range_lbls;
#   end
  
#   return ;
# end
