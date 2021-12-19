import numpy as np
import os
import cv2



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

def get_path_name(category):
    inp_img_path="data/"+category+"/img"
    gt_bb_path="data/"+category+"/groundtruth_rect.txt"
    #pred_bb_path="data/"+category+"/pred_bb_"+method+".txt"
    return (inp_img_path,gt_bb_path)

def bb_iou(pred_bb, gt_bb):
    '''
    Input: pred_bb=coordinates of predicted bounding box
           gt_bb= coordinates of groundtruth bounding box
        Both have dimension (top left corner, right bottom corner=(x1,y1,x2,y2))
    Output:IOU score
    
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(pred_bb[0], gt_bb[0])
    y1 = max(pred_bb[1], gt_bb[1])
    x2 = min(pred_bb[2], gt_bb[2])
    y2 = min(pred_bb[3], gt_bb[3])

    # compute the area of intersection rectangle
    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    if inter_area == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    pred_bb_area =(pred_bb[2] - pred_bb[0]) * (pred_bb[3] - pred_bb[1])
    gt_bb_area = (gt_bb[2] - gt_bb[0]) * (gt_bb[3] - gt_bb[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(pred_bb_area+ gt_bb_area - inter_area)

    # return the intersection over union value
    return iou


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

