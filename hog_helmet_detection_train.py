import numpy as np
import cv2
import glob, time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
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

def getreshapedimg(input_file):
    img=cv2.imread(input_file)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    img = cv2.resize(img, (128, 128))
    return img 


def imagestack(filearray):
    img_stk=[]
    for i in range(len(filearray)):
        img=getreshapedimg(filearray[i])
        img_stk.append(img)
    img_stk= np.asarray(img_stk)
    return img_stk 


def hog_helmet_detection_train(args):
    helmet=sorted(glob.glob(args.image_location_helmet))
    nohelmet=sorted(glob.glob(args.image_location_nohelmet))
    helmet_img, nohelmet_img=[],[]
    helmet_img = imagestack(helmet)
    nohelmet_img = imagestack(nohelmet)
    X = np.vstack((helmet_img, nohelmet_img) )
    y = np.hstack((  np.zeros(len(helmet_img)),np.ones(len(nohelmet_img)) ))
    X_hog=[]
    for i in range(X.shape[0]):
        X_hog.append(hog_features(X[i,:,:]))
    X_hog= np.asarray(X_hog)
    X_hog=np.squeeze(X_hog, axis=2)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y, 
                                                        test_size=args.test_size_helmet,
                                                        random_state=rand_state)

    svc=LinearSVC(max_iter = args.iteration_helmet)
    t1=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print ("Model fitting time for helmet classification: ", round(t2 - t1, 2))
    joblib.dump(svc, args.model_location_helmet) 
    
