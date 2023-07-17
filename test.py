import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
# import numpy as np

# from torch.utils.data import Dataset
# from PIL import Image

from libs.model_definitions import initialize_model
from libs.train_model import train_model


# TODO: missing: dataloaders_dict, class_names


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders_dict['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                img = inputs.cpu().data[j] / 2
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)

                plt.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# ImageNet Constants
mean = [0.485, 0.456, 0.406] 
std =  [0.229, 0.224, 0.225]

# data directory with [train, val, test] dirs
data_dir = "./dataset/views_split"

model_name = "photo-sanitizer"
checkpoint_path = f"checkpoints/{model_name}.pt"

input_size = 224

force_CPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() and not force_CPU else "cpu")

batch_size = 64

# # Remember to first initialize the model and optimizer, then load the dictionary locally.
# model, input_size = initialize_model(model_name, num_classes, train_deep)
# model = model.to(torch.device(device))

# optimizer = optim.Adam(model.parameters(), lr=0.001)

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


model = torch.load(checkpoint_path)
model = model.to(torch.device(device))

# set dropout and batch normalization layers to evaluation mode before running inference
model.eval()


#TEST
# create data loader for test-data
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

test_dir = os.path.join(data_dir, "test")

testset = datasets.ImageFolder(test_dir, test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

test_acc = 0.0
for samples, labels in test_loader:
    with torch.no_grad():
        samples, labels = samples.to(device), labels.to(device)
        output = model(samples)

        # calculate accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(labels)
        test_acc += torch.mean(correct.float())
        
print('Accuracy of the network on {} test images: {}%'.format(len(testset), round(test_acc.item()*100.0/len(test_loader), 2)))

# visualize_model(model)

