#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports Your used libaraies
from collections import OrderedDict
from torchvision import datasets , transforms ,models 
from torch import nn # import neural network from torch library
from torch import optim #import optimizer from torch
import numpy as np # used to make operations on numbers
import pandas as pd 
import torch.nn.functional as F 
import matplotlib.pyplot as plt #used to show the output and any figure 
import seaborn as sb 
import torch 
from PIL import Image


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers' #directory of all flowers images
train_dir = data_dir + '/train' #flowers/train
valid_dir = data_dir + '/valid' #flowers/valid
test_dir = data_dir + '/test'   #flowers/test


# In[3]:


#Transformations made on photos to convert it to Numbers to make operations on it

modified_transforms = transforms.Compose([
                              transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              #Convert our photo to pixels with RGB Values (Red , Blue ,Green)
                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                              #transforms.Normalize(mean,standard deviation)
                             ])


image_datasets = datasets.ImageFolder(data_dir, transform=modified_transforms)#Hold all the data
train_datasets = datasets.ImageFolder(train_dir, transform=modified_transforms)#Hold the train dataset
valid_datasets = datasets.ImageFolder(valid_dir, transform=modified_transforms)#Holds cross-validation dataset
test_datasets = datasets.ImageFolder(test_dir, transform=modified_transforms)  #Hold test dataset

#To take a batch of the photos every time 
#not all the photos entered at the same time
#make shuffle on the photos 
#torch.utils.data.DataLoader is a method takes three arguments (data,batch_size=int,shuffle=boolean )

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json 

#Category to Name 
#cat_to_name holds the name and number of the flower
#i.e. {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue"}
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    #print(len(cat_to_name))
    


# In[5]:


##Creating a  pre-trained model

model=models.vgg16(pretrained=True) #Get a pre-trained model from the internet
print('Before',model.classifier)

for param in model.parameters():
                                #Gradient Descent gets the global optimum "The samllest error"
    param.requires_grad = False #Disables Gradient Descent for each parameter , "The model is already trained" 
                                #No need to change parameters(Weights) 


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)), #Applies Linear Transformation to the incoming data
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 512)),
                          ('relu2',nn.ReLU()),#The operation made between layers
    
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1)) #Takes The value of the o/p and converts it into probabilities 
                          ]))

model.classifier = classifier
#print('After',model.classifier)
criterion=nn.NLLLoss() #Estimates the loss between the predicted output and the real ouptput (y_actual-y_predicted)
optimizer=optim.Adam(model.classifier.parameters(),lr=0.001) #To Choose the learning rate and determine the best accuracy


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[6]:



    
# TODO: Do training on the test set
def test_valid_set(model,testloader,criterion,device='cuda'):
    test_loss=0
    accuracy=0
    for images,labels in testloader:
        images,labels = images.to('cuda'),labels.to('cuda')
        optimizer.zero_grad()
            
        
        output=model.forward(images)
        test_loss+=criterion(output,labels).item()
        ps=torch.exp(output)
        equality=(labels.data==ps.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
        
    return test_loss,int(accuracy)
        
#test_loss,accuracy=test_valid_set(model,testloaders,criterion,'cuda')
#print(test_loss,accuracy) 

# TODO: Build and train your network
def train_network(model,epochs,print_every,trainloader,criterion,optimizer,device='cuda'):
    steps=0
    epochs=epochs
    print_every=print_every
    model.to(device)
    
    for e in range(epochs):
        model.train()
        running_loss=0
        for inputs,labels in (trainloader):
            
            steps+=1 #To print the output every 40 step 
            inputs,labels = inputs.to('cuda'),labels.to('cuda') 
            optimizer.zero_grad()#To make zero changes on the inputs 
            
            #Forward & Backward Pass
            output=model.forward(inputs) #The Predicted output 
            
            loss=criterion(output,labels) #(Y_actual - Y_predicted )
            loss.backward()   #Make backward propagation and reduce the loss by changing the weights
            optimizer.step()  #To move a step forward in the optimizer
        
            
            running_loss+=loss.item() #To add the loss every step
            
            if steps % print_every ==0:
                model.eval()
                with torch.no_grad():
                    test_loss,accuracy=test_valid_set(model,testloaders,criterion,'cuda')
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Train Loss: {:.4f}..".format(running_loss/print_every),
                      "Test Loss :{:.4f}..".format(test_loss/len(testloaders)),
                     "Test Accuracy: {:.4f} %".format(accuracy/len(testloaders) * 100)
                     )

                running_loss = 0 
                model.train()
print('Started training your network..!')               
train_network(model,3,40,trainloaders,criterion,optimizer,'cuda')
print('Done training your network..!')
        
        


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[11]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_element_index = 1 if width > height else 0
    min_element_index= 0 if width> height else 1

    aspect_ratio=picture_coords[max_element_index]/picture_coords[min_element_index]
    new_picture_coords = [0,0]
    new_picture_coords[min_element_index] = 256
    new_picture_coords[max_element_index] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    width, height = new_picture_coords
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / ([255]*3)
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

new_image = process_image('flowers/train/1/image_06734.jpg')


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[13]:


def predict(image_path, mymodel, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
       
        image.unsqueeze_(0)
        image = image.float()
        
        model, _ = load_checkpoint(mymodel)
        outputs = model.forward(image)
        probs, classes = torch.exp(outputs).topk(topk)
        names=list()
   
        ''''''
        classes=classes[0].add(1).tolist()
        #print(classes)
        for i in range(len(classes)):
            names.append(cat_to_name[str(classes[i])])
        
        #print(names)
        return probs[0].tolist(),names
        
    # TODO: Implement the code to predict the class from an image file
    
predict('flowers/train/1/image_06750.jpg','checkpoint.pth')


# In[14]:


# TODO: Display an image along with the top 5 classes
def display_classified_image(classified_image_path):
    plt.style.use('ggplot')
    
    image=process_image(classified_image_path)
    
    
    probs,names=predict(classified_image_path,'checkpoint.pth')
    data={'probs':[round(p,3) for p in probs],'names':[name.title() for name in names]}
    df=pd.DataFrame(data)
    
    
    base_color = sb.color_palette()[0]
    plt.bar(data['names'],data['probs'],color='green',)
    plt.ylim(0,1)
    plt.xticks(data['names'], data['names'],rotation=80)
    plt.yticks(np.arange(0,1.1,.1))
    plt.xlabel('Classes')
    plt.ylabel('Probabilities')
    
    #Showing the Image
    imshow(image)
    
    
display_classified_image('flowers/train/5/image_05161.jpg')
    
    
    


# In[ ]:




