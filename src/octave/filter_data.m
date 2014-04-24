function [projected_labels,projected] = filter_data(x,w,min_proj,max_proj)


%project the test instances and eliminate the filtered out classes
projected = x * w;
nclasses = size(min_proj,1);

projected_labels = ones(size(projected,1),nclasses,size(projected,2));
for c = 1 : nclasses,
    in_range_lbls = ones(size(projected,1),1,size(projected,2));
    in_range_lbls(:,:,1) = projected(:,1) >= min_proj(c,1) &  projected(:,1) <= max_proj(c,1);
    if ( size(projected,2) > 1 )
      for j = 2 : size(projected,2),
          in_range_lbls(:,:,j)  = and(in_range_lbls(:,:,j-1), projected(:,j) >= min_proj(c,j) &  projected(:,j) <= max_proj(c,j));
      end
    end
    projected_labels(:,c,:) = in_range_lbls;
end
