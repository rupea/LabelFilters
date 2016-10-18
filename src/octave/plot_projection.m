function [pd,al] = plot_projection(x,w,min,max,p)

l=min(:,p);
u=max(:,p);
a = l<u;
l=l(a);
u=u(a);
nc = [repmat(1,1,size(l,1)) repmat(-1,1,size(u,1))];
[si,i] = sort([l' u']);
snc = nc(i);
proj = x*w(:,p);
[nn,xx] = hist(proj,100);
pd = [xx' (nn/trapz(xx,nn))'];
al = [si' (cumsum(snc)/size(a,1))'];
[ax,h1,h2]=plotyy(xx,nn/trapz(xx,nn),si,cumsum(snc)/size(a,1))
xlabel("Projection");
ylabel(ax(1),"Fraction of test points");
ylabel(ax(2),"Fraction of active classes");


