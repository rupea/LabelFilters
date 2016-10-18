function plot_projection2(x,w,min,max,w2,min2,max2,p)

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
l2=min2(:,p);
u2=max2(:,p);
a2 = l2<u2;
l2=l2(a2);
u2=u2(a2);
nc2 = [repmat(1,1,size(l2,1)) repmat(-1,1,size(u2,1))];
[si2,i2] = sort([l2' u2']);
snc2 = nc2(i2);
proj2 = x*w2(:,p);
[nn2,xx2] = hist(proj2,100);

plotyy(xx,nn/trapz(xx,nn),si,cumsum(snc)/size(a,1))
hold on
[ax,h1,h2]=plotyy(xx2,nn2/trapz(xx2,nn2),si2,cumsum(snc2)/size(a2,1))

xlabel("Projection");
ylabel(ax(1),"Fraction of test points");
ylabel(ax(2),"Fraction of active classes");


