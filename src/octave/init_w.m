function w= init_w(which_init, x, tr_label,d,proj_dim)

if which_init == 1
  %initialize w with the lda solution -- slow
    [w] = lda(x,tr_label);
    if false
        plot_projections(x,w,tr_label,classes,proj_dim);
    end
elseif which_init == 2
  % randomly initialize w
    w = randn(d,proj_dim)*10/sqrt(d);
end

return ;
