function plot_projections(x,w,tr_label, classes,l,u,x_test, ts_label, exp_name,C1,C2,topdf)
colors = {'b.', 'k.', 'r.', 'm.','b.', 'k.', 'r.', 'm.'};

if ~exist('exp_name', 'var'),
    exp_name= '';
else
    exp_name= ['_' exp_name];
end

if ~exist('topdf', 'var'),
  topdf=1;
end

classes = unique(tr_label)';

colors = repmat(colors,1,ceil(length(classes)/8));
for j=1:size(w,2)
  try
    if (topdf == 1)
      figure('visible', 'off');
    else
      figure()
    end
    
    hold on
    for i = 1 : length(classes)
      c = classes(i);
      tmp=x(tr_label==c,:)*w(:,j);    
      range = (max(tmp)-min(tmp));
      if exist('l','var') && exist('u', 'var')
        x1=linspace(l(c,j),u(c,j),100);
        plot(x1 ,i-.5, colors{i});
      end
      plot(tmp,i, colors{i});
    end
    hold off
    if (topdf == 1),
      filename=sprintf('projected_data_%s_C1_%d_C2_%d_proj_%d_train.pdf', exp_name, C1, C2, j);
      print ('-dpdf', filename);
    end
  catch 
    disp(["error happended: " lasterror.message]);
  end

  try
    if (topdf == 1)
     figure('visible', 'off');
    else
      figure()
    end
    hold on
    for i = 1 : length(classes)
        c = classes(i);
        tmp=x_test(ts_label==c,:)*w(:,j);    
        range = (max(tmp)-min(tmp));
        if exist('l','var') && exist('u', 'var')
            x1=linspace(l(c,j),u(c,j),100);
            plot(x1 ,i-.5, colors{i});
        end
        plot(tmp,i, colors{i});
    end
    hold off
    if (topdf == 1)
      filename=sprintf('projected_data_%s_C1_%d_C2_%d_proj_%d_test.pdf', exp_name, C1, C2, j);
      print ('-dpdf', filename);
    end

  catch 
    disp(["error happended: " lasterror.message]);
  end
end
