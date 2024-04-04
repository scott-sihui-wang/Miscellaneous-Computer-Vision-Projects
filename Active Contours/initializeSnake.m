function [x, y] = initializeSnake(I)

% Show figure
imshow(I)
[h w]=size(I);
step=10*(h+w);

fprintf("\nDo you want to save your control points to file (Y/N)?\n");
while 1
    [~,~,b]=ginput(1);
    if b~='Y' & b~='y' & b~='N' & b~='n'
        fprintf("Error Message: Please input Y or N to indicate if you want to save the control points to file, or the program WILL NOT proceed.\n");
    else
        if b=='Y' | b=='y'
            output_filename=input("Please indicate the filename (.mat):",'s');
        end
        n_pts=0;
        % Get initial points

        fprintf("----------Instructions----------\n");
        fprintf("Click the image with the left button of the mouse to add control points to initialize the snake.\n")
        fprintf("You should specify AT LEAST THREE control points, or the program WILL NOT proceed.\n");
        fprintf("After specifying the control points, please press Q, q, ESC or Spacebar to continue.\n");
        fprintf("If you click outside the image region, the control point will be clamped to stay within the image region.\n");

        while 1
            [input_x input_y,input_b]=ginput(1);
            if (input_b=='q' | input_b=='Q' | input_b==27 | input_b==32) & n_pts > 2
                break;
            else 
                if input_b==1
                    n_pts=n_pts+1;
                    usr_x(n_pts)=min(max(input_x,1),size(I,2));usr_y(n_pts)=min(max(input_y,1),size(I,1));
                    hold on
                    plot(usr_x(n_pts),usr_y(n_pts),'bo','lineWidth',2);
                end
            end
        end
        if b=='Y' | b=='y'
            save("control_points/"+output_filename+".mat",'usr_x','usr_y');
        end
        break;
    end
end

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

