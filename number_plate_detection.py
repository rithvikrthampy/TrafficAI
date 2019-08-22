import cv2
import pytesseract
from config import args
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
def license_plate_reading(args):
    image_01 = cv2.imread(args.image)
    winh = args.winh
    winw = args.winw
    capt_str = []
    for x, y, window in sliding_window(image_01, 
                                       stepSize=args.stepsize, 
                                       windowSize = (winw, winh)):
        if window.shape[0] != winh or window.shape[1] != winw:
            continue
        img_1 = image_01[y:y + winh, x:x + winw]
        cv2.imshow("cropped", img_1)
        cv2.waitKey(1)
        capt_str.append(pytesseract.image_to_string(img_1))
    capt_str = [k.replace('\n', ' ') for k in capt_str]
    capt_str = sorted([k for k in capt_str if k != ''], key = len, reverse = True)
    return capt_str[:args.bestresultscount]

if __name__ == '__main__':
    lic_num = license_plate_reading(args)
    print(lic_num)
    
    
