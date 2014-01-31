function [projected_labels,projected] = \
project_tests(x,tr_label,xtest,test_labels, pr_func, params,exp_name,no_projections,restarts)
  
C1 = params(1);
C2 = params(2);

if strcmp(pr_func,"find_w")==1
    disp('-----------------------------------------------------');
    best_obj=Inf;
    for r=1:restarts
      disp(sprintf('restart %d\n', r));
      w=init_w(2, x, tr_label, size(x,2), no_projections);
    
      [w,min_proj,max_proj, obj_val]=oct_find_w(x,tr_label,C1,C2,w);
      
      if (best_obj > obj_val(length(obj_val)))
        best_w = w;
        best_min_proj = min_proj;
        best_max_proj = max_proj;
        best_obj_val = obj_val;
        best_obj = obj_val(length(obj_val));
      end
    end
     
    w = best_w;
    min_proj = best_min_proj;
    max_proj = best_max_proj;
    obj_val = best_obj_val;

    % plot the objective value vs iteration 	   
    figure('visible','off');
    plot(obj_val, 'b', 'LineWidth', 3);

    print('-dpdf', sprintf('objective_plot_%s_C1_%d_C2_%d.pdf', exp_name,C1, C2))

else
    [w,min_proj,max_proj, obj_val] = feval(pr_func,x,tr_label,params, no_projections);
end

save(sprintf('results/wlu_%s_C1_%d_C2_%d.mat',exp_name, C1, C2), "w", "min_proj", "max_proj" , "obj_val");

%project the test instances and eliminate the filtered out classes
projected = xtest * w;
classes = unique(tr_label)';
ts_labels = zeros(size(xtest,1),length(classes));

plot_projections(x,w,tr_label, classes, min_proj, max_proj,xtest, test_labels, exp_name, C1, C2);

projected_labels = ones(size(projected,1),length(classes));

for c = 1 : length(classes),
    in_range_lbls = ones(size(projected,1),1);
    for j = 1 : size(projected,2),
        in_range_lbls  = and(in_range_lbls, projected(:,j) >= min_proj(c,j) &  projected(:,j) <= max_proj(c,j));
    end
    projected_labels(:,c) = in_range_lbls;
end

return ;
