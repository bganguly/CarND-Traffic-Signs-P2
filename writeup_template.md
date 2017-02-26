#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_for_readme/data_summary.jpg "Basic data summary"
[image2]: ./output_for_readme/label_distribution.jpg "Label distribution"
[image3]: ./output_for_readme/preprocess_before.jpg "Sample image before preprocessing"
[image4]: ./output_for_readme/preprocess_summary.jpg "Sample image pixel summary after preprocessing"
[image5]: ./output_for_readme/preprocess_after.jpg "Sample image after preprocessing"
[image6]: ./output_for_readme/test_run_results.jpg "hyperparameter tuning"
[image7]: ./images_from_web/no_entry_cropped.jpg "no entry"
[image8]: ./images_from_web/pedestrians_cropped.jpg "pedestrians "
[image9]: ./images_from_web/right_of_way_cropped.jpg "right of way"
[image10]: ./images_from_web/road_work_cropped.jpg "road work"
[image11]: ./images_from_web/speed_limit_60_cropped.jpg "speed limit 60"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the required README.  
All code cells mentioned here are in the IPython notebook located in [Traffic_Signs_Recognition.ipynb](https://github.com/bganguly/CarND-Traffic-Signs-P2/blob/master/Traffic_Signs_Recognition.ipynb).  
For brevity, i will just refer to the IPython notebook as simply 'notebook'

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cells 3,4 and 5 of the  notebook.

Cell 3 loads the train/test pickle files and makes appropriate train/test features/labels available to the rest of the notebook.  
Cell 4 than further one-hot encodes the train/test labels and makes those  available to the rest of the notebook.  
Cell 5 does the actual statistical aggregate reporting.  
Here is a summary of the data set. 

![alt text][image1]

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cell 6 of the  notebook.  
We calculate and then Plot the number of features corresponding to a given label.

Here is an exploratory visualization of the data set. 
![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell 7 of the  notebook.  
The data has been run through -  
- np.mean(axis=3) - this squelches the data to 32X32 from 32x32x3
- normalize_greyscale() - this transforms the 0-255 to 0.1-0.9 . 

The np.mean() reduces a 3 element array to the average value of the array, and helps with further downstream processing.  
The normalize_grescale is an attamept to coalesce widely spread values to a narrow band, so that the classification can be done in realistic time and so that it converges to a value with respectable accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]
![alt text][image5]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the code cell 8 of the  notebook.  
To cross validate my model, I randomly split the training data into a training set and validation set.  
I did this by using sklear.train_test_split()

My final training set had 90%  of originl training images and My validation set had 10% of images.  
The number of test images did not change.  

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for the final model architecture is contained in the code cell 9 of the  notebook.  

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for the final model architecture is contained in the code cell 9 and 12 of the  notebook.  

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with default hyperparameters of 30 epochs/64 as batch_size and x/=255 as the data sqleching. Then over a series of test varied these. Noted the validation accuracy/loss as well as the test accuracy/loss. The goal is to get the highest accuracy / least loss possible.
Results are below 
- https://drive.google.com/file/d/0B9vOjB65N3QkOUhfMElydUZ0MEE/view?usp=sharing . 
A screenshot of the various parameter tuned is in cell 10 of the  notebook.   
![alt text][image6]


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Five random German traffic signs were collected online and roughly cropped to be about square. each image was a jpeg. Then code was written to resize these to be 32x32x3 each.

Here are five German traffic signs that I found on the web, and are listed in cell 14 of the notebook:

![alt text][image7] 
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

There is an image of a pedestrian walking that might not match exactly  in the sample set supplied, so maybe that would be off. All other are pretty crisp to start with and should generally classified to the 85% accuracy or so reported by the test set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for the predictions is contained in the code cell 15 and 16 of the  notebook.  

Here are the results of the prediction:

['images_from_web/no_entry_cropped.jpg', 'images_from_web/speed_limit_60_cropped.jpg', 'images_from_web/right_of_way_cropped.jpg', 'images_from_web/pedestrians_cropped.jpg', 'images_from_web/road_work_cropped.jpg']
[[ 0.385  0.351  0.051  0.041  0.04 ]
 [ 0.571  0.129  0.066  0.035  0.034]
 [ 0.505  0.093  0.065  0.062  0.047]
 [ 0.221  0.163  0.162  0.142  0.107]
 [ 0.84   0.097  0.016  0.013  0.01 ]]
[[20 21 33 41 22]
 [16 41 38 22 12]
 [41 22 26 33 16]
 [18 16 33  8 22]
 [22 16 38 21 42]]

Based on the above there appears to be a rather poor classification of images obtained from the web.
The above indicates that the following was inferred (using signnames.csv)-
- 'no_entry_cropped.jpg'-> 20:General Caution
- 'speed_limit_60_cropped.jpg'-> 16:Vehicles over 3.5 metric tons prohibited
- 'right_of_way_cropped.jpg'-> 41:End of no passing
- 'pedestrians_cropped.jpg'-> 18:General Caution 
- 'road_work_cropped.jpg'-> 22:Bumpy Road  

So, we have essentially a 0% accuracy on the images obtained from the web, as opposed to the earlier obtained 85.6% accuracy on the validation set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for the prediction certainty is in the code cell 17 of the  notebook.  

[[ 0.367  0.21   0.155  0.062  0.059]
 [ 0.374  0.271  0.091  0.07   0.063]
 [ 0.297  0.287  0.102  0.079  0.069]
 ..., 
 [ 0.753  0.081  0.058  0.044  0.019]
 [ 0.738  0.078  0.073  0.04   0.017]
 [ 0.506  0.24   0.067  0.047  0.04 ]]
[[28 16 29 20 15]
 [16 28 29 15 42]
 [16 28 29 15 42]
 ..., 
 [16 15 28 42 29]
 [16 15 29 42 28]
 [16 15 28 42  9]]

Based on the above there appears to be a rather poor **certainty** of predictions, as the probablities are almost uniformly distributed.
