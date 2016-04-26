function write_svm_data(filename, y_tr, x_tr, y_te=[], x_te=[])
  if (any(size(y_tr) == 1))   % multiclass problem
    tr_file = sprintf("%s.train.svm",filename);
    libsvmwrite(tr_file,y_tr,x_tr);
    if (!isempty(x_te))
      te_file = sprintf("%s.test.svm",filename);
      libsvmwrite(te_file,y_te,x_te);
    endif
  else %multilabel problem
    out = fopen([filename ".train.svm"],"w");
    for i=1:size(y_tr,1)
      fprintf(out,"%s",strjoin(strsplit(num2str(find(y_tr(i,:)))),","));
      fprintf(out, " %d:%g",[find(x_tr(i,:)); x_tr(i,find(x_tr(i,:)))]);
      fprintf(out,"\n");
    end
    fclose(out);
    if (!isempty(x_te))
      out = fopen([filename ".test.svm"],"w");
      for i=1:size(y_te,1)
	fprintf(out,"%s",strjoin(strsplit(num2str(find(y_te(i,:)))),","));
	fprintf(out, " %d:%g",[find(x_te(i,:)); x_te(i,find(x_te(i,:)))]);
	fprintf(out,"\n");
      end
      fclose(out);
    endif
  endif
  
end
      