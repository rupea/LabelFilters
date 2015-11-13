function [] = plot_by(x, y=[], group=[],legend={})
  
  if (!isempty(group))
    levels = unique(group);
%    figure;
    hold on;
    c = -1;
    i=0;
    for l=levels'
      i=i+1;
      if (isempty(legend))
	key=num2str(l);
      else
	key=legend{i};
      endif
      c = c+1;
      if (c == 5)
	c=c+1;
      endif
      if (!isempty(y))
	plot(x(group==l),y(group==l),["-" num2str(mod(c,6) + 1) ";" key ";"]);
      else
	plot(x(group==l),["-" num2str(c)]);
      endif
    endfor
    hold off
  else
    if (!isempty(y))
      plot(x,y);
    else
      plot(x);
    endif
  endif
end