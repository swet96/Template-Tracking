import os  
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import bb_iou,get_path_name,get_frames
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat')

args = parser.parse_args()

category = args.cat

#category="Liquor" # can take values from "BlurCar2", "Bolt", "Liquor"

def get_pred_coords(image,template,template_shape,method):
    '''
    Input: image: input image where the object to be detected
           template: the template image which is to be detected
           template_shape:(width,height) of the template, note: width=no of columns
           method: which template matching method to be used, e.g., cv2.TM_SQDIFF 
    Output: coords: cordinates of the bounding box, 
                    (top left corner,bottom right corner)=(x1,y1,x2,y2)
    '''
    w=template_shape[0]    
    h=template_shape[1]
    res = cv2.matchTemplate(image,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method==cv2.TM_SQDIFF:
        x1,y1,x2,y2 = min_loc[0],min_loc[1],min_loc[0]+w,min_loc[1]+h #TODO
    else:
        x1,y1,x2,y2 = max_loc[0],max_loc[1],max_loc[0]+w,max_loc[1]+h #TODO
    coords=(x1,y1,x2,y2)    
    return coords

#method to draw rectabgle in teh image given the co-ordinates
def get_bb(image,coords):
    ''' 
    Input: image: image where bounding box is to be drawn
           coords: coordinates of the bounding box 
           (top left corner, bottom right corner)=(x1,y1,x2,y2)
    Output: None
    '''
    x1,y1,x2,y2=coords
    top_left=(x1,y1)
    bottom_right=(x2,y2)
    cv2.rectangle(image,top_left, bottom_right, 255, 2)


def run_model(category,method):
    inp_img_path,gt_bb_path=get_path_name(category)
    frames,coords=get_frames(inp_img_path,gt_bb_path)
    init_image=cv2.imread(os.path.join(inp_img_path,frames[0])) 
    init_coords=coords[0]
    x,y,w,h=init_coords
    template=init_image[y:y+h,x:x+w]
    template_shape=(w,h)

    eval_frames=frames[1:] #TODO frames[1:,]
    #each row is of (x,y,w,h) format
    gt_coords=coords[1:] #TODO coords[1:,]

    iou_array=np.array([])
    gt_bb_array=np.array([])
    pred_bb_array=np.array([])
    count=0

    for frame,coords in zip(eval_frames,gt_coords):
        count+=1
        if count%100==0:
            print(frame)
        eval_frame=cv2.imread(os.path.join(inp_img_path,frame))  
        #predicted bounding box coordinates  
        pred_bb=get_pred_coords(eval_frame,template,template_shape,method)
        
        x,y,w,h=coords
        #coordinates of the groundtruth bounding box 
        gt_bb=(x,y,x+w,y+h)
        iou=bb_iou(pred_bb,gt_bb)
        iou_array=np.append(iou_array,iou)
        gt_bb_array=np.append(gt_bb_array,gt_bb)
        pred_bb_array=np.append(pred_bb_array,pred_bb)
        x1,y1,x2,y2=pred_bb
        template=eval_frame[y1:y2,x1:x2]
        template_shape=template.shape
        #print(template_shape)
    pred_bb_array=pred_bb_array.reshape(-1,4)
    gt_bb_array=gt_bb_array.reshape(-1,4)
    return iou_array,pred_bb_array,gt_bb_array



method=cv2.TM_SQDIFF
print(f"Category is:{category} ")
print("SSD: ")
iou_ssd,pred_bb_ssd,gt_bb_ssd=run_model(category,method)
pred_bb_ssd=pred_bb_ssd.astype(int)
gt_bb_ssd=gt_bb_ssd.astype(int)
#print("SSD score is: "+str(np.mean(iou_ssd)))


method=cv2.TM_CCORR_NORMED
print("NCC: ")
iou_ncc,pred_bb_ncc,gt_bb_ncc=run_model(category,method)
pred_bb_ncc=pred_bb_ncc.astype(int)
gt_bb_ncc=gt_bb_ncc.astype(int)
print("Done!")

def get_values(pred_bb_array,gt_bb_array):
    pred_bb_array=pred_bb_array.reshape(-1,4)
    gt_bb_array=gt_bb_array.reshape(-1,4)
    iou_array=np.array([])
    mod_pred_bb_array=np.array([])
    for pred_bb,gt_bb in zip(pred_bb_array,gt_bb_array):
        #print(pred_bb.shape)
        x1,y1,x2,y2=pred_bb
        x,y,w,h=x1,y1,x2-x1,y2-y1
        x,y,w,h=x,y,h,w
        pred_bb=x,y,x+w,y+h
        mod_pred_bb_array=np.append(mod_pred_bb_array,pred_bb)
        iou_array=np.append(iou_array,bb_iou(pred_bb,gt_bb))
    return iou_array,mod_pred_bb_array.reshape(-1,4)



iou_ncc,pred_bb_ncc=get_values(pred_bb_ncc,gt_bb_ncc)
iou_ssd,pred_bb_ssd=get_values(pred_bb_ssd,gt_bb_ssd)

pred_bb_ssd=pred_bb_ssd.astype(int)
pred_bb_ncc=pred_bb_ncc.astype(int)

print(f"Mean mIOU for NCC method for category {category} is: {np.round(np.mean(iou_ncc),3)}")
print(f"Mean mIOU for SSD method for category {category} is: {np.round(np.mean(iou_ssd),3)}")