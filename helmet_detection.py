import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
helmet_detection_model = joblib.load('models/helmet_detection_model.pkl')


def hog_features(img):
    winSize = (128,128)
    blockSize = (16,16)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    hist = hog.compute(img)
    return hist


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5)):


    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def find_helmet_windows(img, classifier, y_start_stop=[0, 480], xy_window=(128, 128), xy_overlap=(0.85, 0.85) ):
    helmet_windows = []
    windows = slide_window(img, y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap)
    for window in windows:
        img_window = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (128, 128))
        img_window=cv2.cvtColor(img_window,cv2.COLOR_RGB2GRAY)
        features =  np.squeeze(hog_features(img_window))
        pred = int(classifier.predict(features.reshape(1, -1)))
        if pred == 0:
            helmet_windows.append(window)
    return helmet_windows


def draw_helmet_windows(img, windows):
   
    output = np.copy(img)
    return draw_boxes(output, windows)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=2):

    img_copy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(img_copy, bbox[0], bbox[1], color, thick)
    return img_copy


def add_heat(heatmap, bbox_list):
  
   
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1


    return heatmap

def apply_threshold(heatmap, threshold):


    heatmap[heatmap <= threshold] = 0

    return heatmap

def draw_labeled_bboxes(img, labels):

    box=[]
    for helmet_number in range(1, labels[1]+1):

        nonzero = (labels[0] == helmet_number).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        box.append(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)

    return img, box



def draw_labels_on_helmet(img, boxes, threshold = 3):
 
    heatmap = add_heat(np.zeros(img.shape), boxes)
    heatmap_thresholded = apply_threshold(heatmap, threshold)
    labels = label(heatmap_thresholded)
    return draw_labeled_bboxes(np.copy(img), labels)


def process_manager(img_pro,
                    x = 128,
                    y = 128,
                    scale = [1, 1.25, 1.5],
                    threshold = 11):
    temp=[]
    for sc in scale:
        temp.append(find_helmet_windows(img_pro, 
                                      helmet_detection_model, 
                                      y_start_stop=[0, img_pro.shape[0]], 
                                      xy_window=(int(sc*x), 
                                      int(sc*y))))
    windows = []
    for box_set in temp:
        for box in box_set:
            windows.append(box)
    box_img = draw_helmet_windows(img_pro, windows)
    img_helmet, helmet_box = draw_labels_on_helmet(img_pro, 
                                             windows,
                                             threshold = threshold)
    return box_img, img_helmet, helmet_box

def helmet_detetion_pro(img_pro, args):
    img = cv2.cvtColor(img_pro, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    features =  np.squeeze(hog_features(img))
    pred = int(helmet_detection_model.predict(features.reshape(1, -1)))
    if pred == 0:
        return 0
    else:
        return None
