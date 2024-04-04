clc;
clear;
close all;

background_filename=input("Please input the filename of the background image:(for example: ./test.jpg)\n",'s');
im_background = im2double(imread(background_filename));

n_blend=input("Please specify the number of objects to blend in:\n");
for i=1:n_blend
    
    [h_b,w_b,n_b]=size(im_background);
    if n_b~=1 && n_b~=3
        fprintf("Error: Please check if valid images are provided.");
        quit(-1);
    end
    
    target_filename=input("Please input the filename of the image of object No. "+i+":\n",'s');
    im_object = im2double(imread(target_filename));
    
    [h_o,w_o,n_o]=size(im_object);
    if n_o~=1 && n_o~=3
        fprintf("Error: Please check if valid images are provided.");
        quit(-1);
    end
    
    if n_b > n_o
        im_temp=zeros(h_o,w_o,n_b);
        for i=1:n_b
            im_temp(:,:,i)=im_object(:,:,min(i,n_o));
        end
        im_object=im_temp;
    else if n_o > n_b
            im_temp=zeros(h_b,w_b,n_o);
            for i=1:n_o
                im_temp(:,:,i)=im_background(:,:,min(i,n_b));
            end
            im_background=im_temp;
        end
    end
    
    objmask = get_mask(im_object);
    [im_s, mask_s] = align_source(im_object, objmask, im_background);

    disp('start');
    im_blend = poisson_blend(im_s, mask_s, im_background);
    disp('end');
    imshow(im_blend);
    im_background=im_blend;
end

output_filename=input("Please specify the filename of the output image:\n",'s');

imwrite(im_blend,output_filename);
figure(), hold off, imshow(im_blend);