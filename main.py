import bike_detection
import helmet_detection
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import glob
d = sorted(glob.glob("/home/anup/02_work/bits/helmet/data/bike_class/bike/*.*"))


for i in d:
    if bike_detection.bike_detetion_pro(cv2.imread(i)) == 0:
        helmet_code = helmet_detection.helmet_detetion_pro(cv2.imread(i))
        if helmet_code != 0:
            print("\n\n")
            print (i.split('/')[-1])
            print ("Bike detected")
            print("No helmets detected")
        else:
            print("\n\n")
            print (i.split('/')[-1])
            print(helmet_code)
            print("Helmets detected")

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
print(pytesseract.image_to_string(Image.open('/home/anup/Pictures/crop1.jpg')))


