clc;
clear;
close all;

imgin = im2double(imread('./large.jpg'));

[imh, imw, nb] = size(imgin);
assert(nb==1);
% the image is grayscale

V = zeros(imh, imw);
V(1:imh*imw) = 1:imh*imw;
% V(y,x) = (y-1)*imw + x
% use V(y,x) to represent the variable index of pixel (x,y)
% Always keep in mind that in matlab indexing starts with 1, not 0

%TODO: initialize counter, A (sparse matrix) and b.
%-----
k=1;
sz=5*(imw-2)*(imh-2)+6*(imw-2)+6*(imh-2)+4;
i=zeros(sz,1);
j=zeros(sz,1);
v=zeros(sz,1);
b=zeros(imh*imw,1);
%-----

%TODO: fill the elements in A and b, for each pixel in the image
%-----
for q=2:imw-1
    i(k)=V(1,q);j(k)=V(1,q);v(k)=2;k=k+1;    
    i(k)=V(1,q);j(k)=V(1,q-1);v(k)=-1;k=k+1;
    i(k)=V(1,q);j(k)=V(1,q+1);v(k)=-1;k=k+1;
    b(V(1,q))=2*imgin(1,q)-imgin(1,q-1)-imgin(1,q+1);
    i(k)=V(imh,q);j(k)=V(imh,q);v(k)=2;k=k+1;
    i(k)=V(imh,q);j(k)=V(imh,q-1);v(k)=-1;k=k+1;
    i(k)=V(imh,q);j(k)=V(imh,q+1);v(k)=-1;k=k+1;
    b(V(imh,q))=2*imgin(imh,q)-imgin(imh,q-1)-imgin(imh,q+1);
end
for p=2:imh-1
    i(k)=V(p,1);j(k)=V(p,1);v(k)=2;k=k+1;    
    i(k)=V(p,1);j(k)=V(p-1,1);v(k)=-1;k=k+1;
    i(k)=V(p,1);j(k)=V(p+1,1);v(k)=-1;k=k+1;
    b(V(p,1))=2*imgin(p,1)-imgin(p-1,1)-imgin(p+1,1);
    i(k)=V(p,imw);j(k)=V(p,imw);v(k)=2;k=k+1;
    i(k)=V(p,imw);j(k)=V(p-1,imw);v(k)=-1;k=k+1;
    i(k)=V(p,imw);j(k)=V(p+1,imw);v(k)=-1;k=k+1;
    b(V(p,imw))=2*imgin(p,imw)-imgin(p-1,imw)-imgin(p+1,imw);
end
for p=2:imh-1
    for q=2:imw-1
        i(k)=V(p,q);j(k)=V(p,q);v(k)=4;k=k+1;
        i(k)=V(p,q);j(k)=V(p-1,q);v(k)=-1;k=k+1;
        i(k)=V(p,q);j(k)=V(p+1,q);v(k)=-1;k=k+1;
        i(k)=V(p,q);j(k)=V(p,q-1);v(k)=-1;k=k+1;
        i(k)=V(p,q);j(k)=V(p,q+1);v(k)=-1;k=k+1;
        b(V(p,q))=4*imgin(p,q)-imgin(p-1,q)-imgin(p+1,q)-imgin(p,q-1)-imgin(p,q+1);
    end
end
%-----

%TODO: add extra constraints
%-----
UL=[0 1 1 0 0];
UR=[0 1 0 0 0];
BL=[0 1 1 1 0];
BR=[0 1 0 1 1];
for cnt=1:5
    UpperLeft=UL(cnt);UpperRight=UR(cnt);BottomLeft=BL(cnt);BottomRight=BR(cnt);
    i(k)=V(1,1);j(k)=V(1,1);v(k)=1;k=k+1;
    if UpperLeft==true
        b(V(1,1))=1;
    else
        b(V(1,1))=imgin(1,1);
    end
    i(k)=V(1,imw);j(k)=V(1,imw);v(k)=1;k=k+1;
    if UpperRight==true
        b(V(1,imw))=1;
    else
        b(V(1,imw))=imgin(1,imw);
    end
    i(k)=V(imh,1);j(k)=V(imh,1);v(k)=1;k=k+1;
    if BottomLeft==true
        b(V(imh,1))=1;
    else
        b(V(imh,1))=imgin(imh,1);
    end
    i(k)=V(imh,imw);j(k)=V(imh,imw);v(k)=1;k=k+1;
    if BottomRight==true
        b(V(imh,imw))=1;
    else
        b(V(imh,imw))=imgin(imh,imw);
    end
    
    A=sparse(i,j,v);
    
    k=k-4;

%-----


%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
    solution = A\b;
    error = sum(abs(A*solution-b));
    disp(error)
    imgout = reshape(solution,[imh,imw]);
    filename=sprintf('output%d.png',cnt);
    imwrite(imgout,filename);
    figure(), hold off, imshow(imgout);
end



