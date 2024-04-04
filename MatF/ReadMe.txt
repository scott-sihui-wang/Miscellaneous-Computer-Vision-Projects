Environment Configurations:

Ubuntu 20.04 LTS
Python 3.7.11
Numpy 1.20.3
OpenCV 4.5.4-dev

Run Normalized 8-point Algorithm:

Open the terminal in Ubuntu system, input the commands:
python FM.py
python FM.py --image1=data/left1.jpg --image2=data/right1.jpg
python FM.py --image1=data/view1.jpg --image2=data/view2.jpg
to see the images of results for each of the 3 pair of images.

The code will save the images of the results. The filename will be "8PTS_result_01_"+filename of the original image1+".png" and "8PTS_result_02_"+filename of the original image2+".png"

To compare the numerical results with cv2's implementation, open the file "FM.py" and uncomment the last 3 lines in the function: FM_by_normalized_8_point
Then open the terminal and input the above mentioned commands. This time you can see the numerical results of the fundamental matrices.
Go to the folder "8PTS_Numerical" to see the screenshots of the numerical results.

For comparison, we have generated and saved the results of cv2's 8 points algorithm. The filename starts with "CV2_8PTS_result".


Run RANSAC to compute fundamental matrices:

Open the terminal in Ubuntu system, input the commands:
python FM.py --UseRANSAC=1
python FM.py --UseRANSAC=1 --image1=data/left1.jpg --image2=data/right1.jpg
python FM.py --UseRANSAC=1 --image1=data/view1.jpg --image2=data/view2.jpg
to see the images of results for each of the 3 pair of images

The code will save the images of the results. The filename will be "RANSAC_result_01_"+filename of the original image1+".png" and "RANSAC_result_02_"+filename of the original image2+".png"

For comparison, we have generated and saved the results of cv2's RANSAC algorithm. The filename starts with "CV2_RANSAC_result".

