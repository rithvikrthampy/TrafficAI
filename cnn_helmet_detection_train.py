import torch
import numpy as np
import time
from cnn_model import Net
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision import models


def cnn_model_train(args):
       train_transforms = transforms.Compose([transforms.Resize(( \
                                           args.cnn_helmet_resize_width, 
                                           args.cnn_helmet_resize_height)),
                                           transforms.ToTensor()])
    

    train_data = datasets.ImageFolder(args.dataset_location,
                                      transform=train_transforms)
    

    train_loader = data_utils.DataLoader(train_data, 
                                         batch_size=args.batch_size, 
                                         shuffle=True,
                                         drop_last = True)
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr = args.learning_rate) 
    
    for epoch in range(args.epoch):  
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch no: %d, Epoch loss: %.3f, Epoch time = %.3f' %
              (epoch + 1, running_loss, time.time() - start_time))
    print('Finished Training')
    
    torch.save(model.state_dict(), 
               args.cnn_model_location)
    
