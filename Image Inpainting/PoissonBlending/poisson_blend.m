function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s);
[outh, outw, ~]=size(im_t);

imgout=zeros(outh,outw,nb);

sz_mask=sum(sum(mask_s));
location=zeros(sz_mask,2);
index=zeros(imh,imw);
cnt=0;

for i=1:imh
    for j=1:imw
        if mask_s(i,j)~=0
            cnt=cnt+1;
            index(i,j)=cnt;  
            location(cnt,1)=i;
            location(cnt,2)=j;
        end
    end
end

%TODO: consider different channel numbers
for ch=1:nb
    b=zeros(sz_mask,1);
    x=0;
    y=0;
    v=0;
    k=1;
    for l=1:sz_mask
        if location(l,1)>1
            if index(location(l,1)-1,location(l,2))~=0
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                x(k)=l;y(k)=index(location(l,1)-1,location(l,2));v(k)=-1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1)-1,location(l,2),ch);
            else
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1)-1,location(l,2),ch)+im_t(location(l,1)-1,location(l,2),ch);
            end
        end
        if location(l,1)<imh
            if index(location(l,1)+1,location(l,2))~=0
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                x(k)=l;y(k)=index(location(l,1)+1,location(l,2));v(k)=-1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1)+1,location(l,2),ch);
            else
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1)+1,location(l,2),ch)+im_t(location(l,1)+1,location(l,2),ch);
            end
        end
        if location(l,2)>1
            if index(location(l,1),location(l,2)-1)~=0
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                x(k)=l;y(k)=index(location(l,1),location(l,2)-1);v(k)=-1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1),location(l,2)-1,ch);
            else
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1),location(l,2)-1,ch)+im_t(location(l,1),location(l,2)-1,ch);
            end
        end
        if location(l,2)<imw
            if index(location(l,1),location(l,2)+1)~=0
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                x(k)=l;y(k)=index(location(l,1),location(l,2)+1);v(k)=-1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1),location(l,2)+1,ch);
            else
                x(k)=l;y(k)=l;v(k)=1;k=k+1;
                b(l)=b(l)+im_s(location(l,1),location(l,2),ch)-im_s(location(l,1),location(l,2)+1,ch)+im_t(location(l,1),location(l,2)+1,ch);
            end
        end
    end
    A=sparse(x,y,v);
    solution = A\b;
    error = sum(abs(A*solution-b));
    disp(error)
    imgout(:,:,ch)=im_t(:,:,ch);
    for p=1:sz_mask
        imgout(location(p,1),location(p,2),ch)=solution(p);
    end
end
%TODO: initialize counter, A (sparse matrix) and b.
%Note: A don't have to be k¡Ák,
%      you can add useless variables for convenience,
%      e.g., a total of imh*imw variables
%-----
%-----

%TODO: fill the elements in A and b, for each pixel in the image
%-----
%-----

%TODO: add extra constraints (if any)
%-----
%-----


%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents


%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
%-----
%-----
