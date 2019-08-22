import torch
from cnn_model import Net
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category= UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = Net()
if torch.cuda.is_available():
    model.cuda()

def helmet_detection(img, args):
    model.load_state_dict(torch.load(args.cnn_model_location))
    predict_transforms = transforms.Compose([transforms.Resize(( \
                                           args.cnn_helmet_resize_width, 
                                           args.cnn_helmet_resize_height)),
                                           transforms.ToTensor()])
    image_tensor = predict_transforms(Image.open(img)).float()
    out = model(image_tensor.view(1,3,args.cnn_helmet_resize_width,
                                  args.cnn_helmet_resize_height))
    return (int(torch.argmax(out.cpu())))



