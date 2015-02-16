function [projected_labels,projected] = \
      project_tests_multilabel(x,tr_label,xtest, parameters, exp_name,restarts, resume)

  %% learn the projection
  plot_objval = true; % generate a plot of the objective values
  [w, max_proj, min_proj] = learn_projections(x,tr_label,parameters,exp_name,restarts,resume, plot_objval);
  
  disp(norm(w));
  for j=1:size(w,2)-1
    for k=j+1:size(w,2)
      fprintf(1,"W%d * W%d = %f \n", j,k,w(:,j)'*w(:,k)/(norm(w(:,j))*norm(w(:,k))));
    end
  end
  
  %% project the test data and get the mask for the filtered labels
  if (size(tr_label, 2) == 1)
    %% multiclass problem with label vector 
    %% assumes taht labels are 1:noLabels
    noLabels = max(tr_label);
  else
    %% multi label or multiclass problem with sparse label matrix
    noLabels = size(tr_label,2);
  end

  [projected_labels, projected] = project_data(xtest, noLablels, w, min_proj, max_proj);
  
end
