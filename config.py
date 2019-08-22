import argparse
parser = argparse.ArgumentParser()

# =============================================================================
# Arguments for HOG bike/nobike classifier
# =============================================================================
parser.add_argument('--image-location-bike', default='data/bike_class/bike/*.*',
                    help='location of the images for bike class')
parser.add_argument('--image-location-nobike', default='data/bike_class/nobike/*.*',
                    help='location of the images for nonbike class')
parser.add_argument('--test-size-bike', default=0.0,
                    help='designate the size of test dataset for bike/nobike classification.')
parser.add_argument('--iteration-bike', default=10000,
                    help='number of iteration of the SVC classifier')
parser.add_argument('--model-location-bike', default='models/bike_detection_model.pkl',
                    help='location where the bike detection model needs to be saved.')

# =============================================================================
# Arguments for HOG helmet/nohelmet classifier
# =============================================================================
parser.add_argument('--image-location-helmet', default='data/helmet_class/helmet/*.*',
                    help='location of the images of helmet class')
parser.add_argument('--image-location-nohelmet', default='data/helmet_class/nohelmet/*.*',
                    help='location of the images of non-helmet class')
parser.add_argument('--test-size-helmet', default=0.0,
                    help='designate the size of test dataset for helmet/nohelmet classification')
parser.add_argument('--iteration-helmet', default=10000,
                    help='number of iteration of the SVC classifier for helmet classification.')
parser.add_argument('--model-location-helmet', default = 'models/helmet_detection_model.pkl',
                    help='location where the model needs to be saved.')

# =============================================================================
# Arguments for CNN helmet/nohelmet classifier
# =============================================================================
parser.add_argument('--cnn-helmet-resize-width', default=128,
                    help='width of the resize window for CNN classifier')
parser.add_argument('--cnn-helmet-resize-height', default=128,
                    help='height of the resize window for CNN classifier')
parser.add_argument('--dataset-location', default='data/helmet_class',
                    help='folder containing the train and test dataset of CNN classifier')
parser.add_argument('--batch-size', default=4,
                    help='size of the batch for CNN classifier')
parser.add_argument('--learning-rate', default=0.001,
                    help='learning rate of the CNN classifier')
parser.add_argument('--momentum', default=0.9,
                    help='momentum of the SGD classifier')
parser.add_argument('--epoch', default=100,
                    help='number of epochs of the classifier')
parser.add_argument('--cnn-model-location', default='models/cnn_helmet.pt',
                    help='location to save CNN model')

# =============================================================================
# Tesseract License Plate detection routine
# =============================================================================
parser.add_argument('--winh', default=24,
                    help='height of the sliding window')
parser.add_argument('--winw', default=128,
                    help='width of the sliding window')
parser.add_argument('--stepsize', default=12,
                    help='step size for the sliding window')
parser.add_argument('--bestresultscount', default=5,
                    help='step size for the sliding window')

# =============================================================================
# Argument for image location
# =============================================================================
parser.add_argument('--image', default='',
                    help='location of the image')

# =============================================================================
# Arguments for model to be used for helmet detection
# =============================================================================
parser.add_argument('--helmet-model', default='cnn',
                    help='Helmet detection model CNN or HOG')

# =============================================================================
# Arguments for training required
# =============================================================================

args = parser.parse_args()

