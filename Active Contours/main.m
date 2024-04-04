clear all;
clc;
cla;
while 1
    fprintf("----------Snake: Active Contour Program----------\n");
    % Parameters (play around with different images and different parameters)
    N = 500;
    alpha = 1.0;
    beta = 1.0;
    gamma = 1.0;
    kappa = 1.0;
    Wline = -1.0;
    Wedge = 1.0;
    Wterm = 1.0;
    sigma = 0.5;
    file_name=["images/circle.jpg" "images/square.jpg" "images/shape.png" "images/star.png" "images/brain.png" "images/dental.png" "images/vase.tif"];
    ctrl_pts=["control_points/circle.mat" "control_points/square.mat" "control_points/shape.mat" "control_points/star.mat" "control_points/brain_outer_shell.mat" "control_points/dental.mat" "control_points/vase.mat" "control_points/brain_inner_contour.mat" "control_points/brain_right_eye.mat"];
    name=["circle" "square" "shape" "star" "brain_outer_shell" "dental" "vase" "brain_inner_contour" "brain_right_eye" "brain_custom"];
    params=[100.0 10.0 5.0 0.2 -8.0 1.0 1.0 0.5 60;
            3.0 3.0 0.45 1.0 -1.0 1.0 1.0 2.0 160;
            0.8 0.8 0.8 1.1 0.8 2.0 2.0 2.0 160;
            0.1 50.0 1.0 1.0 -3.0 3.0 0.0 2.2 150;
            20.0 20.0 1.0 1.0 1.0 1.0 1.0 1.5 200;
            1.0 20.0 1.0 1.0 -1.0 8.0 0.0 1.5 200;
            1.0 1.0 1.0 1.0 -1.0 1.0 0.0 1.5 100;
            0.0 1000.0 1.0 1.0 -20.0 20.0 0.0 2.5 250;
            1.0 1.0 1.0 1.0 -1.0 1.0 1.0 1.5 50];
    fprintf("----------Image Selection----------\n");
    fprintf("1. circle\n2. square\n3. shape\n4. star\n5. brain\n6. dental\n7. vase\n");
    % Load image
    while 1
        [~,~,b_f]=ginput(1);
        if b_f>='1' && b_f<='7'
            I=imread(file_name(b_f-'0'));
            break;
        else
            fprintf("Error Message: Please input a number between 1 and 7 to specify the image, or the program WILL NOT proceed.\n");
        end
    end

    if (ndims(I) == 3)
        I = rgb2gray(I);
    end
    I=im2double(I);
    imshow(I);

% Initialize the snake

    fprintf("----------Initialize the Snake: Adding Control Points----------\n");
    fprintf("1. Demo Mode: Initialize the snake by loading predefined control points and predefined parameters.\n2. Manual Mode: You need to manually add the control points to initialize the snake. In this mode, different images share the same set of parameters.\n");
    index_modifier=0;
    while 1
        [~,~,b_m]=ginput(1);
        if b_m=='1'
            if b_f~='5'
                [x, y] = initializeSnakeDemo(I,ctrl_pts(b_f-'0'));
            else
                fprintf("For the brain MRI image, please further specify which is your desired control points:\n1. Outer shell of skull\n2. Inner contour of the brain matter\n3. The right eye hole\n");
                while 1 
                    [~,~,sel]=ginput(1);
                    if sel=='1'
                        [x, y] = initializeSnakeDemo(I,ctrl_pts(b_f-'0'));
                        break;
                    else if sel=='2' | sel=='3'
                            [x, y] = initializeSnakeDemo(I,ctrl_pts(sel+6-'0'));
                            index_modifier=sel-'0'+1;
                            break;
                        else
                            fprintf("Error Message: Please input a number 1 between 3 to specify the control points, or the program WILL NOT proceed.\n");
                        end
                    end
                end
            end        
            break;
        else if b_m=='2'
                if b_f=='5'
                    index_modifier=5;
                end
                [x, y] = initializeSnake(I);
                break;
            else
                fprintf("Error Message: Please input a number 1 or 2 to specify the mode, or the program WILL NOT proceed.\n");
            end
        end
    end

    saveas(gcf,"init_snakes\init_"+name(b_f-'0'+index_modifier)+".png");

    % Calculate external energy
    if b_m=='1'
        alpha=params(b_f-'0'+index_modifier,1);
        beta=params(b_f-'0'+index_modifier,2);
        gamma=params(b_f-'0'+index_modifier,3);
        kappa=params(b_f-'0'+index_modifier,4);
        Wline=params(b_f-'0'+index_modifier,5);
        Wedge=params(b_f-'0'+index_modifier,6);
        Wterm=params(b_f-'0'+index_modifier,7);
        sigma=params(b_f-'0'+index_modifier,8);
        N=params(b_f-'0'+index_modifier,9);
    end
    I_smooth = double(imgaussfilt(I, sigma));
    Eext = getExternalEnergy(I_smooth,Wline,Wedge,Wterm);

    % Calculate matrix A^-1 for the iteration

    fprintf("Please Specify the Internal Function:\n1. The pre-compiled built-in function\n2. The self-implemented bonus function\n");
    while 1
        [~,~,sel]=ginput(1);
        if sel=='1'
            Ainv = getInternalEnergyMatrix(size(x,2), alpha, beta, gamma);
            break;
        else if sel=='2'
                Ainv = getInternalEnergyMatrixBonus(size(x,2),alpha,beta,gamma);
                break;
            else
                fprintf("Error Message: Please input a number 1 or 2 to specify the function, or the program WILL NOT proceed.\n");
            end
        end
    end

    % Iterate and update positions
    displaySteps = floor(N/10);
    fprintf("\n----------Snake Iterations----------\n");
    for i=1:N
        % Iterate
    
        [x,y] = iterate(Ainv, x, y, Eext, gamma, kappa);

        % Plot intermediate result
        imshow(I); 
        hold on;
        plot([x x(1)], [y y(1)], 'r','lineWidth',2);
        
        % Display step
        if(mod(i,displaySteps)==0)
            fprintf('%d/%d iterations\n',i,N);
        end
    
        pause(0.0001)
    end

    %saveas(gcf,"contour_results\contour_"+name(b_f-'0'+index_modifier)+".png");
 
    if(displaySteps ~= N)
        fprintf('%d/%d iterations\n',N,N);
    end
    h=fill(x,y,[1 1 0]);
    set(h,'facealpha',0.6);

    %saveas(gcf,"segmentation_results\seg_"+name(b_f-'0'+index_modifier)+".png");
    
    fprintf("Press Spacebar to continue, press ESC to quit.\n");
    key=0;
    while 1
        [~,~,key]=ginput(1);
        if key==32 | key==27
            break;
        end
        fprintf("Error: Invalid input. Please press Spacebar to continue, press ESC to quit.\n");
    end
    if key==32
        figure,hold off;
        continue;
    else
        break;
    end
end
