#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/left.jpg "Recovery Image"
[image4]: ./examples/right.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_new.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model_new.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 6 convolution neural network layers, 2 dropout and 4 dense layers. (model_new.py lines 168-185).

The model includes RELU layers to introduce nonlinearity (code line 173&176&178), and the data is normalized in the model using a Keras lambda layer (code line 169). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model_new.py lines 173&176&178). 

I use the three directions data to train the model. The left and right directions have a angle 0.25 to revise so that they have the same angle as the right ahead. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_new.py line 187).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And I also collect the data by running the car in both clockwise and anti-clockwise. When I find the car run out of the lan, I will train more in some spacial place such as some corners and the bridge.
When I train the model, I find that the data I collected is not balanced. So I balance them first. I split 1000 parts from 0 to 1. Then choose at most 500 samples which steering locate in corresponding part.   
With thses data, I process them first. When I train the model, I find that the car could not run smoothly in some shadow regions, so I add some shadow randomly on the images. And also add a GaussianBlur on them. By these way, the model becomes robust.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA's model. I thought this model might be appropriate because the NVIDIA's model is built for self driving car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the size of each layer and add some dropout so that it can run robust.

Then I add a kernel_regularizer of regularizers.l2 in each dense layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the trackto improve the driving behavior in these cases, I drive the car in these places many more times.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model_new.py lines 163-180) consisted of a convolution neural network with the following layers.

Here is a visualization of the architecture: 

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive in the middle of the road.

![alt text][image3]   ![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I just run the car in both clockwise and anti-clockwise, thinking that this would make the model more robust.


After the collection process, I had 4147 number of data points. I then preprocessed this data by these ways. First I balance the data to avoid the influnce of some too centralized data. Then I improve the gamma value of the images and add some shadow and also some blur on the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
