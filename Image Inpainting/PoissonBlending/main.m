clc;
clear;
close all;

im_background = im2double(imread('./background.jpg'));
im_object = im2double(imread('./target.jpg'));

% get source region mask from the user
objmask = get_mask(im_object);

% align im_s and mask_s with im_background
[im_s, mask_s] = align_source(im_object, objmask, im_background);

% blend
disp('start');
im_blend = poisson_blend(im_s, mask_s, im_background);
disp('end');
imshow(im_blend);

im_background=im_blend;
im_object = im2double(imread('./test1.jpg'));
objmask = get_mask(im_object);
[im_s, mask_s] = align_source(im_object, objmask, im_background);

disp('start');
im_blend = poisson_blend(im_s, mask_s, im_background);
disp('end');

imwrite(im_blend,'output.png');
figure(), hold off, imshow(im_blend);
