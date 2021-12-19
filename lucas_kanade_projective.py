import cv2
import numpy as np
import os
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat')

args = parser.parse_args()

category = args.cat

#category="Liquor" # can take values from "BlurCar2", "Bolt", "Liquor"

def iou(first_bb, second_bb):
    # An example of first bounding box
    img0 = cv2.imread(os.path.join(inp_img_path,frames[0]), 0)
    first_bb = [[int(num) for num in sub] for sub in first_bb]
    second_bb = [[int(num) for num in sub] for sub in second_bb]

    first_bb_points = first_bb #[[250, 210], [440, 210], [440, 390], [250, 390]]
    stencil = np.zeros(img0.shape).astype(img0.dtype)
    contours = [np.array(first_bb_points)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result1 = cv2.bitwise_and(img0, stencil)
    result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)

    # An example of second bounding box
    second_bb_points = second_bb #[[280, 190], [438, -190], [438, 390], [280, 390]]
    stencil = np.zeros(img0.shape).astype(img0.dtype)
    contours = [np.array(second_bb_points)]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)
    result2 = cv2.bitwise_and(img0, stencil)
    result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)

    # IoU calculation
    intersection = np.logical_and(result1, result2)
    union = np.logical_or(result1, result2)
    iou_score = np.sum(intersection) / np.sum(union)
    #print('IoU is %s' % iou_score)
    return iou_score

def get_frames(inp_img_path,gt_bb_path):
    ''' 
    Input: inp_img_path: relative path of the file where the image sequences are present
           gt_bb_path: path where the coordinates of the groundtruth
                       bounding box are present, coordinates of the form 
                       (top left corner, bottom right corner)
    Output: frames: array containing file names 
            coords: array of dim (no of images, 4)
    '''
   
    #co-ordinates of the ground truth bounding box format (x,y,h,w)
    coords=np.loadtxt(gt_bb_path,dtype=int)
    #frames are extracted
    frames=np.array(sorted(os.listdir(inp_img_path)))   
    return frames,coords



inp_img_path,gt_bb_path=get_path_name(category)
frames,coords=get_frames(inp_img_path,gt_bb_path)

eval_frames=frames[1:]
eval_coords=coords[1:]


def get_image_grad(image):
    ksize=3
    gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    return gX,gY

def get_del_p(template_bb,warp_image_gray,template_gray,W):
    (x1,y1,x2,y2)=template_bb
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    w=x2-x1
    h=y2-y1

    warp_image_gray_cut=warp_image_gray[y1:y2,x1:x2]
    gX_cut,gY_cut=get_image_grad(warp_image_gray_cut)
    hessian=0
    other_term=0
    
    diff = warp_image_gray_cut - template_gray
    rows = diff.shape[0]
    cols = diff.shape[1]

    gX_cut_1D = gX_cut.reshape(-1,1)
    gY_cut_1D = gY_cut.reshape(-1,1)

    del_I_del_W_del_p = np.concatenate((gX_cut_1D, gY_cut_1D), axis=1)

   
    steepest_descent = np.empty((diff.shape[0]*diff.shape[1],9))    #TODO
 
    k=0
    for ix,iy in np.ndindex(diff.shape):
        pixel_grad = del_I_del_W_del_p[k]
        
        const=-(pixel_grad[0]+pixel_grad[1])/np.power(W[2,0]*ix+W[2,1]*iy+W[2,2],2)
        
        steepest_descent[k] = [ix*pixel_grad[0], iy*pixel_grad[0], pixel_grad[0],  ix*pixel_grad[1], iy*pixel_grad[1], pixel_grad[1],ix*const,iy*const,const] 
        k += 1
    
    hessian = np.zeros((steepest_descent.shape[1],steepest_descent.shape[1]))
    for row in steepest_descent:
          hessian = np.add( np.dot(row.T,row), hessian)
    diff = diff.reshape(-1,1).ravel()

    other_term = steepest_descent * diff[:, np.newaxis]

    other_term = np.sum(other_term,axis=0)


    del_p=np.linalg.pinv(hessian)@other_term
    return del_p

def get_mat(a11, a12, a13, a21, a22, a23,a31,a32,a33):
    trans_mat=np.float32([[ a11, a12, a13],[a21, a22, a23],[a31,a32,a33]])
    return trans_mat


init_image=cv2.imread(os.path.join(inp_img_path,frames[0])) 
init_coords=coords[0]
x,y,w,h=init_coords
template=init_image[y:y+h,x:x+w]
template_bb=(x,y,x+w,y+h)
template_gray=cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

ious = []
pred_bb_arr=np.array([])
gt_bb_arr=np.array([])

count=0
start=time.time()
print(f"Lucas-Kanade Projective for category:{category}")
for eval_frame,eval_coord in zip(eval_frames,eval_coords):
    count+=1
    if count%100==0:
        print("Time taken till now:",np.round(time.time()-start,2))
        print(eval_frame)

    eval_image=cv2.imread(os.path.join(inp_img_path,eval_frame)) 
    eval_image_gray=cv2.cvtColor(eval_image, cv2.COLOR_RGB2GRAY)
    x,y,w,h=eval_coord
    eval_gt_bb=(x,y,x+w,y+h)

    
    a11,a12,a13,a21,a22,a23,a31,a32,a33=1,0,0,0,1,0,0,0,1
    W=get_mat(a11, a12, a13, a21, a22, a23,a31,a32,a33)

    eps=1e-3
    error=1
    itr = 1

    while(error>eps and itr<=5):
        warp_image_gray= cv2.warpPerspective(eval_image_gray, W, (eval_image.shape[1], eval_image.shape[0]))
        del_p=get_del_p(template_bb,warp_image_gray,template_gray,W)

        a11 +=del_p[0]
        a12 +=del_p[1]
        a13 +=del_p[2] 
        a21 +=del_p[3]
        a22 +=del_p[4]
        a23 +=del_p[5]
        a31 +=del_p[6]
        a32 +=del_p[7]
        a33 +=del_p[8]
        

        error=np.linalg.norm(del_p)

        W=get_mat(a11, a12, a13, a21, a22, a23,a31,a32,a33)

        itr += 1



    template_bb=W@np.array([[x,x, x+h,x+h], [y,y+w, y+w, y], [1,1,1,1]])
    top_left_x = min(template_bb[0,:])
    top_left_y = min(template_bb[1,:])
    bot_right_x = max(template_bb[0,:])
    bot_right_y = max(template_bb[1,:])
    template_bb = (top_left_x, top_left_y, bot_right_x, bot_right_y)

    ious.append( iou( [[eval_gt_bb[0], eval_gt_bb[1]], [eval_gt_bb[0], eval_gt_bb[3]],[eval_gt_bb[2], eval_gt_bb[3]], [eval_gt_bb[2], eval_gt_bb[1]]],[[template_bb[0],
    template_bb[1]], [template_bb[0], template_bb[3]],[template_bb[2], template_bb[3]], [template_bb[2], template_bb[1]]]) )
    template_gray = eval_image_gray[int(top_left_y):int(bot_right_y),int(top_left_x):int(bot_right_x)]



print(f"The mIOU for the Lucas-Kanade parametrized Algorithm for the Affine case is:{mean(ious)} ")