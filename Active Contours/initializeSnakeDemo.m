function [x, y] = initializeSnakeDemo(I,mat)

% Show figure
imshow(I)
[h w]=size(I);
step=10*(h+w);

fprintf("----------Loading predefined control points----------\n")
pts=load(mat);

usr_x=pts.usr_x;usr_y=pts.usr_y;
n_pts=length(usr_x);
hold on
plot(usr_x,usr_y,'bo','lineWidth',2)

% Interpolate

fprintf("----------Initialize the Snake: Generate the Spline Interpolation Curve----------\n");

ctrl_theta=0:2*pi/n_pts:2*pi;
ctrl_pts=[usr_x usr_x(1);usr_y usr_y(1)];
pp=spline(ctrl_theta,ctrl_pts);
yy=ppval(pp,linspace(0,2*pi,step+1));
k=1;x(k)=floor(yy(1,1));y(k)=floor(yy(2,1));
for i=1:10*(h+w)
    if floor(yy(1,i+1))~=floor(yy(1,i)) || floor(yy(2,i+1))~=floor(yy(2,i))
        k=k+1;x(k)=floor(yy(1,i+1));y(k)=floor(yy(2,i+1));
    end
end

% Clamp points to be inside of image
x=min(max(x,1),w);
y=min(max(y,1),h);

hold on
plot(x,y,'r','lineWidth',2)

end

