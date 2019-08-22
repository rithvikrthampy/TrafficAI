import bike_detect
import helmet_detect
from tkinter import*
from tkinter import scrolledtext
from tkinter import messagebox
import cv2
import number_plate_detection
from config import args
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

f=open("output.txt",'a+')
def process_bike_helmet(args):
    bike_status = bike_detect.detect_bike(args)
    if bike_status == 1:
        print("Bike Detected")
		
         
        if args.helmet_model == 'hog':
            helmet_status = helmet_detection.helmet_detetion_pro(cv2.imread(args.image), args)
        elif args.helmet_model == 'cnn':
            helmet_status = helmet_detect.pred_helmet(args)
        else:
            print("Models not working for helmet detection.")
        if helmet_status == 0:
            print("Helmet detected.")
            lic_num = number_plate_detection.license_plate_reading(args)
            print("Detected License numbers:")
        elif helmet_status == 1:
            lic_num = number_plate_detection.license_plate_reading(args)
            print("No helmet detected.")
            print("Detected License numbers:")
            for i in range(len(lic_num)):
               print(i, '. ', lic_num[i])
               tr(i)+'. '+str(lic_num[i])+'\n'
               f.write(temp)
    elif bike_status == 1:
        print("No bike detected.")
    else:
        print("Status: ", bike_status)
        print("There is something wrong with the models!")

if __name__=='__main__':
	process_bike_helmet(args)

       


