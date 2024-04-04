'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0 )
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
args = parser.parse_args()

print(args)

# normalization of the input points

# input variables:
# pts1 2D input points
# in this scenario, they are usually the 2D points from the left image
# pts2 2D input points
# in this scenario, they are usually the 2D points from the right image
# and points in pts2 are supposed to be corresponding points to pts1

# output variables:
# pts1_n: pts1 after normalization.
# pts1_n are 3D points, meaning that (x,y) are represented as (x,y,1)
# pts2_n: pts2 after normalization.
# pts2_n are 3D points, meaning that (x,y) are represented as (x,y,1)
# N1 the transformation matrix used to normalize pts1
# N2 the transformation matrix used to normalize pts2
# The transformation matrices, N1 and N2, are returned, because we still
# need them to "denormalize" the fundamental matrices

def normalization(pts1,pts2):
    assert(pts1.shape[0]==pts2.shape[0]) # first, make sure that pts1 and pts2 contain same number of points
    sz=pts1.shape[0] # the variable sz is the number of point pairs
    x1=pts1[:,0] # x1:collect all x-coordinations for points in the left image
    x2=pts2[:,0] # x2:collect all x-coordinations for points in the right image
    y1=pts1[:,1] # y1:collect all y-coordinations for points in the left image
    y2=pts2[:,1] # y2:collect all y-coordinations for points in the right image
    x1_avg=np.average(x1) # x1_avg: the average of x-coordinations for points in the left image
    x2_avg=np.average(x2) # x2_avg: the average of x-coordinations for points in the right image
    y1_avg=np.average(y1) # y1_avg: the average of y-coordinations for points in the left image
    y2_avg=np.average(y2) # y2_avg: the average of y-coordinations for points in the right image
    m1=np.average(np.power((x1-x1_avg)**2+(y1-y1_avg)**2,0.5)) # this is to calculate the average distance of points in pts1 to their centroid, (x1_avg,y1_avg)
    m2=np.average(np.power((x2-x2_avg)**2+(y2-y2_avg)**2,0.5)) # this is to calculate the average distance of points in pts2 to their centroid, (x2_avg,y2_avg)
    # below is to construct the transformation matrices to normalize the points
    m1=math.sqrt(2)/m1
    m2=math.sqrt(2)/m2
    # build the transformation matrices according to the formula
    N1=np.array([[m1,0,-x1_avg*m1],[0,m1,-y1_avg*m1],[0,0,1]]) # N1 is the transformation matrix for pts1
    N2=np.array([[m2,0,-x2_avg*m2],[0,m2,-y2_avg*m2],[0,0,1]]) # N2 is the transformation matrix for pts2
    # to perform normalization using N1 and N2, we need to represent pts1 and pts2 in homogeneous coordinations
    # to represent 2D point (x,y) in homogeneous coordinations, we just need to convert it to 3D coordination (x,y,1)
    pts1_aug=np.append(pts1,np.ones((sz,1)),axis=1) # pts1_aug is 3 dimensional, it is pts1 in homogeneous coordination
    pts2_aug=np.append(pts2,np.ones((sz,1)),axis=1) # pts2_aug is 3 dimensional, it is pts2 in homogeneous coordination
    # actual normalization
    pts1_n=pts1_aug@N1.T # normalization of pts1, the output, pts1_n, is 3D in homogeneous coordination
    pts2_n=pts2_aug@N2.T # normalization of pts2, the output, pts2_n, is 3D in homogeneous coordination
    return pts1_n, pts2_n,N1,N2 # return the normalized points (in homogeneous coordinations, and also return the transformation matrices)

# implementation of normalized 8-point algorithm
# input variables:
# pts1: feature points in the left image
# pts2: feature points in the right image
# output variable:
# F: Fundamental matrix calculated by normalized 8-point algorithm

def FM_by_normalized_8_point(pts1,  pts2):
    sz=pts1.shape[0] # get the number of feature points in the left image
    assert(sz>=8) # to check if there are at least 8 points in the left image
    
    # normalization of pts1 and pts2 (implemented in a separate function defined above: normalization)
    pts1_n,pts2_n,N1,N2=normalization(pts1,pts2) # after normalization, get the normalized points pts1_n, pts2_n, and the transformation matrices N1, N2, used for normalization.

    # construct the coefficient matrix of the linear system to solve fundamental matrices
    sol_mat=np.vstack(np.array([pts2_n[i,0]*pts1_n[i,0],pts2_n[i,0]*pts1_n[i,1],pts2_n[i,0],pts2_n[i,1]*pts1_n[i,0],pts2_n[i,1]*pts1_n[i,1],pts2_n[i,1],pts1_n[i,0],pts1_n[i,1],1]) for i in range(sz))

    # mathematically, the solution to the above linear system should be the right singular vector corresponding to the minimal singular value
    u,d,v=np.linalg.svd(sol_mat) # perform singular value decomposition (SVD)
    v=v.T # transpose the right singular vectors
    # after transposition, the 9th column, v[:,8], is the right singular vector corresponding to the minimal singular value.
    # We reorganize v[:,8] as a 3 by 3 matrix, this is the initial solution for fundamental matrices
    F=np.reshape(v[:,8],(3,3))
    
    # Next, we do some refinement for the solution of fundamental matrices
    # What we should do is to impose the constraint that fundamental matrices should have ranks 2
    u,d,v=np.linalg.svd(F) # To achieve this, we perform SVD for the initial solution F
    F=(u@(np.diag(np.array([d[0],d[1],0]))))@v # then we simply set the minimal singular value to zero and recompute the refined solution F
    
    # now, F is for the normalized points, we still need to "denormalize" F to get the solution for the original input point pairs
    F=(N2.T@F)@N1 # "denormalization" of F to get the solution for the original input point pairs

    # Rescale F so that F[2,2]=1 (the element in the 3rd row and in the 3rd column is equal to 1)
    F=F/F[2,2]

    # test: in order to test the accuarcy of this algorithm, uncomment the following 3 lines. # F is obtained by this implementation, G is obtained by cv2 library, by printing them out we can make a comparison
    
    #print(F)
    #G, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    #print(G)
    return  F

# compute the reprojection error of a pair of point, pts1 and pts2, regarding the fundamental matrix F
# input variables:
# pts1: the 2D point in the left image 
# pts2: the 2D point in the right image
# F: the fundamental matrix
# output variable:
# err: the reprojection error

# theoretical explanation and background:
# In epipolar geometry, we have the epipoler constraint, x'Fy=0
# So, if pts1 and pts2 are true matching points, and F is the ground truth fundamental matrix
# we should have: (pts1)' * F * pts2 = 0
# and (pts2)' * F * pts1 = 0
# however, in reality, feature extraction, feature matching, and the calculation of fundamental matrices can all cause errors
# so, we can't expect that the epipolar constraints (pts1)' * F * pts2 = 0 and (pts2)' * F * pts1 = 0 are exactly met
# to compromise a little bit, we want:
# the point in the left image, pts1, is near to the epipolar line F * pts2, and
# the point in the right image, pts2, is near to the epipolar line F * pts1
# if we denote pts1's distance to F * pts2 by err1
# and we denote pts2's distance to F * pts1 by err2
# then we want both err1 and err2 to be small, in order to confirm that the point pair are "inliers" to the model F
# so we define max(err1,err2) as the reprojection error
# and we want the reprojection error to be small, in order to confirm that the point pair, pts1 and pts2, are inliers to the model F

def reproj_err(pts1, pts2, F):
    
    # convert both pts1 and pts2 to 3D homogeneous coordination
    pts1=np.append(pts1,1)
    pts2=np.append(pts2,1)
    
    # compute the epipolar lines
    v1=F@(pts1.T) # this is the epipolar line "induced" by pts2, and pts1 is supposed to fall within certain range of it
    v2=F@(pts2.T) # this is the epipolar line "induced" by pts1, and pts2 is supposed to fall within certain range of it

    # compute the reprojection error
    d1=pts2@v1
    d2=pts1@v2
    err1=d1*d1/(v1[0]*v1[0]+v1[1]*v1[1]) # this is pts2's squared distance to epipolar line F * pts1
    err2=d2*d2/(v2[0]*v2[0]+v2[1]*v2[1]) # this is pts1's squared distance to epipolar line F * pts2
    err=max(err1,err2) # the maximum of two distances is the reprojection error
    return err
    
# implementation of Computing Fundamental Matrices using RANSAC
# input variables:
# pts1: 2D input points in the left image
# pts2: 2D input points in the right image
# output variables:
# F: fundamental matrix computed by RANSAC
# mask: indicating which point is an inlier and which point an is outlier
# mask is a vector composed of "0" and "1". mask[index]=0 means that pts1[index] and pts2[index] are outliers, and mask[index]=1 means that pts1[index] and pts2[index] are inliers

def FM_by_RANSAC(pts1,  pts2):
    sz=pts1.shape[0] # obtain the number of points in pts1
    print(sz)
    assert(sz>=8 and pts2.shape[0]==sz) # make sure that pts1 and pts2 are of the same size, and there are at 8 point pairs
    threshold=50.0 # this is the threshold of reprojection error, the point pair with reprojection error lower than this threshold is considered inliers to the model F
    iter_max=20000 # this is the maximal number of iterations
    num_inliers=0 # initialize the variable "num_inliers". This variable is supposed to store the number of inliers for the best model so far.
    inlier=[] # initialize the variable "inlier". This variable is supposed to store the inliers of the best model so far.
    index=np.arange(0,sz) # the array variable "index" stores all indexs of the point pairs. In each iteration, a sampler randomly select 8 indexes from this array
    for i in range(iter_max):
        if(i%(iter_max/100)==0): # this is to show the progress info of computation
            print("Progress: ",i/(iter_max/100),"percent completed.")
        # randomly select 8 point pairs
        sample=np.random.choice(index,8) 
        p1=pts1[sample]
        p2=pts2[sample]
        # Given 8 point pairs, compute the fundamental matrix, F_cur, by normalized 8-point algorithm
        F_cur=FM_by_normalized_8_point(p1,p2) # F_cur is the fundamental matrix generated by currently selected random point pairs
        # below is to compute which are inliers to F_cur, and how many are F_cur's inliers. This is to evaluate how good is our current model, F_cur
        cnt=0 # initialize the variable "cnt". cnt is supposed to store the number of inliers for the current model, F_cur
        t_mask=np.zeros((sz,1)) # initialize the variable "t_mask". t_mask is supposed to store the information which points are inliers to F_cur (marked by 1) and which points are outliers to F_cur (marked by 0)
        t_inlier=[] # initial the variable "t_inlier". t_inlier is supposed to store the indexes of all inliers to F_cur

        # traverse all the point pairs and compute reprojection errors to see how many inlier pairs are there to the current model F_cur
        for j in range(sz):
            if(reproj_err(pts1[j,:],pts2[j,:],F_cur)<threshold): # if reprojection error is small enough, we mark it as an inlier pair
                cnt=cnt+1 # number of inlier +1
                t_mask[j]=1 # mark the corresponding index j
                t_inlier=np.append(t_inlier,j) # add j to the set of inliers
        # now, we have computed the number of inliers for the current model, F_cur
        # cnt is the number of inliers for the current model, F_cur, and num_inlier is the number of inliers for the best-so-far model, F_best (for the first iteration, F_best is None and num_inlier=0)
        if(cnt>num_inliers and cnt>=8): # if the current model F_cur outperforms the best-so-far model F_best
            print(cnt)
            # then F_cur now is the best-so-far model, below we update the information for best-so-far model
            num_inliers=cnt # F_cur's inliers are best-so-far model's inliers
            F_best=F_cur # F_cur is the best-so-far model
            mask=t_mask # also update the variable "mask" and "inlier" for the best-so-far model
            inlier=t_inlier
    # Now we obtained the best model that we can get from iter_max iterations
    if (len(inlier)>0): # we will use the array "inlier" to find all the inlier point pairs in pts1 and pts2. To avoid type error, we need to transform the elements in the array "inlier" to the type "integer".
        inlier=inlier.astype('int')
    # find all inlier point pairs: pts1[inlier] and pts2[inlier];
    # recompute fundamental matrix given pts1[inlier] and pts2[inlier] as inliers.
    # compared with using F_best directly, this recomputation is likely to achieve better estimation of fundamental matrices
    F=FM_by_normalized_8_point(pts1[inlier],pts2[inlier]) # obtain the final result of fundamental matrix computed by RANSAC

    # the below 4 lines are for comparison purpose. it compares the numerical result obtained by the above implementation and the result obtained by cv2 library.
    # print(F)
    # G, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )
    # print(sum(mask))
    # print(G)
    return  F, mask

img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)  

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F,  mask = FM_by_RANSAC(pts1,  pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]	
else:
    F = FM_by_normalized_8_point(pts1,  pts2)
	

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)

# save the results of the images
# generating the filename
if args.UseRANSAC:
    method='RANSAC'
else:
    method='8PTS'
filename1=args.image1.split(".")[0]
filename1=filename1.split("/")[1]
#plt.savefig('./CV2_8PTS_result_01_'+filename1+'.png')
#plt.savefig('./CV2_RANSAC_result_01_'+filename1+'.png')
plt.savefig('./'+method+'_result_01_'+filename1+'.png')
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)

# generate the filename and save the results of the images
filename2=args.image2.split(".")[0]
filename2=filename2.split("/")[1]
#plt.savefig('./CV2_8PTS_result_02_'+filename2+'.png')
#plt.savefig('./CV2_RANSAC_result_02_'+filename2+'.png')
plt.savefig('./'+method+'_result_02_'+filename2+'.png')
plt.show()
