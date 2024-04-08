# Miscellaneous Computer Vision Projects

This is a collection of assignments from the course of CMPT 732: `Visual Computing Labs I`. The assignments include:

- **Active Contours:** Use energy forces and constraints to extract the boundaries of `RoI` (Region of Interest).

  _As is shown in the demo, the initial curve (left) will recursively shrink to the true boundary of `RoI` (right)._

<img src="/demo/init_curve.png" alt="drawing" width="200"/> <img src="/demo/segmentation.png" alt="drawing" width="200"/>

- **Image Inpainting:** This is an implementation of `Poisson blending` to seamlessly blend images together by `gradient domain fusion` techniques.

  _As is shown below, Possion blending generated much more natural blending of images._

  ![Poisson Blending](/demo/blending.png)

- **Homography Estimation:** Extract matching point pairs from stereo images, and then use two methods (`eight point method` and `RANSAC`) to estimate `homography` matrix.

  _The comparison shows that our implementation (top) can produce results similar to OpenCV's built-in functions (bottom);_

  _RANSAC (right) produces far more accurate results than eight point method (left)._

  ![Homography estimation](/demo/Homography.png)

- **Coons Patch:** This is to generate `coons patch` surfaces from `Bezier curves`. Then, the generated surface is rendered using `Blender`.

  _Below shows the initial Beizer curves, the interpolated coons patch surface, and the rendered surface, respectively._
  
  ![Blender](/demo/blender.png)

For `active contours` and `image inpainting`, please refer to my report [here](report.pdf);

For `homography estimation`, please refer to [readMe](/MatF/ReadMe.txt) for a full explanation of environment requirements, file organization, and command line codes to run the program;

For `coons Patch`, please refer to [readMe](/CoonsPatch/ReadMe.txt) for a full explanation of environment requirements, file organization, and command line codes to run the program.

**Topics:** _Computer Vision_, _Active Contours_

**Skills:** _OpenCV_, _Blender_, _Python_, _Matlab_
