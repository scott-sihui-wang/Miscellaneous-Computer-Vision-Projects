function [newX, newY] = iterate(Ainv, x, y, Eext, gamma, kappa)

% Get fx and fy
D_x=[0 0 0; 0 -1 1; 0 0 0];
D_y=[0 0 0; 0 -1 0; 0 1 0];
f_x=conv2(Eext,D_x,'same');
f_y=conv2(Eext,D_y,'same');
N=length(x);
[h w]=size(Eext);
fx=zeros(N,1);
fy=zeros(N,1);

for i=1:N
    int_x=floor(x(i));
    int_y=floor(y(i));
    if x(i)==int_x && y(i)==int_y
        fx(i)=f_x(int_y,int_x);
        fy(i)=f_y(int_y,int_x);
    else
        if x(i)==int_x && y(i)~=int_y
            fx(i)=(int_y+1-y(i))*f_x(int_y,int_x)+(y(i)-int_y)*f_x(int_y+1,int_x);
            fy(i)=(int_y+1-y(i))*f_y(int_y,int_x)+(y(i)-int_y)*f_y(int_y+1,int_x);
        else
            if x(i)~=int_x && y(i)==int_y
                fx(i)=(int_x+1-x(i))*f_x(int_y,int_x)+(x(i)-int_x)*f_x(int_y,int_x+1);
                fy(i)=(int_x+1-x(i))*f_y(int_y,int_x)+(x(i)-int_x)*f_y(int_y,int_x+1);
            else
                fx(i)=[int_x+1-x(i) x(i)-int_x]*[f_x(int_y,int_x) f_x(int_y+1,int_x); f_x(int_y,int_x+1) f_x(int_y+1,int_x+1)]*[int_y+1-y(i); y(i)-int_y];
                fy(i)=[int_x+1-x(i) x(i)-int_x]*[f_y(int_y,int_x) f_y(int_y+1,int_x); f_y(int_y,int_x+1) f_y(int_y+1,int_x+1)]*[int_y+1-y(i); y(i)-int_y];
            end
        end
    end 
end

% Iterate
newX=(Ainv*(gamma*x'+kappa*fx))';
newY=(Ainv*(gamma*y'+kappa*fy))';

% Clamp to image size
newX=min(max(newX,1),w);
newY=min(max(newY,1),h);

end

