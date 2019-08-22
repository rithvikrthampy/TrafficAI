import bike_detection
import helmet_detection
import cv2
import cnn_helmet_prediction
import number_plate_detection
from config import args
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def process_bike_helmet(args):
    bike_status = bike_detection.bike_detetion_pro(cv2.imread(args.image), args)
    if bike_status == 0:
        print("Bike detected.")
        if args.helmet_model == 'hog':
            helmet_status = helmet_detection.helmet_detetion_pro(cv2.imread(args.image), args)
        elif args.helmet_model == 'cnn':
            helmet_status = cnn_helmet_prediction.helmet_detection(args.image, args)
        else:
            print("Models not working for helmet detection.")
        if helmet_status == 0:
            print("Helmet detected.")
            lic_num = number_plate_detection.license_plate_reading(args)
            print("Detected License numbers:")
            for i in range(len(lic_num)):print(i, '. ', lic_num[i])
        elif helmet_status == 1:
            lic_num = number_plate_detection.license_plate_reading(args)
            print("No helmet detected.")
            print("Detected License numbers:")
            for i in range(len(lic_num)):print(i, '. ', lic_num[i])
    elif bike_status == 1:
        print("No bike detected.")
    else:
        print("Status: ", bike_status)
        print("There is something wrong with the models!")
        
if __name__ == '__main__':
    process_bike_helmet(args)
    
