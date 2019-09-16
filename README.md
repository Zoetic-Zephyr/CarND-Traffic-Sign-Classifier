## Traffic Sign Recognition

### Zheng(Jack) Zhang jack.zhang@nyu.edu

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/before_after.png "Normalize + Grayscale"
[image3]: ./writeup_images/custom_image_combined.png "All Test Traffic Signs"
[image4]: ./custom_images/00-Speed-limit-20km-h.png "Traffic Sign 1"
[image5]: ./custom_images/13-Yield.png "Traffic Sign 2"
[image6]: ./custom_images/34-Turn-left-ahead.png "Traffic Sign 3"
[image7]: ./custom_images/39-Keep-left.png "Traffic Sign 4"
[image8]: ./custom_images/40-Roundabout-mandatory.png "Traffic Sign 5"
[image9]: ./writeup_images/custom_image_out.png "Model Prediction on Custom Images"
[image10]: ./writeup_images/top_five.png "Top Five Predictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Zoetic-Zephyr/CarND-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I randomly select one image from each of the unique classes and display the image as well as the name of that class. You can also see the size of each image equals exactly to 32. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to  I normalized the image data in order to gain a mean zero and equal variance of my image data.

As a last step, convert the images to grayscale because I don't want my model be subjective to different lighting, shadow condition, etc. I want it to minimize those influences, and let the model consider the inner relationship revealed in the architecture of the image.

Here is an example of an original traffic sign image and the augmented image (normalization + grayscaling).

![alt text][image2]



I did not generate more data for training, but this can be an improvement in the future because I discovered that the image classes have different number of training data. Hence, the derived model may be biased.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 |
| ReLU				| rectified linear unit |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
|      ReLU       |            rectified linear unit            |
|   Max pooling   |         2x2 stride,  outputs 5x5x6          |
|     Dropout     |            rectified linear unit            |
| Fully connected |            400 input, 120 output            |
|      ReLU       |            rectified linear unit            |
| Fully connected | 120 input, 84 output |
| ReLU | rectified linear unit |
| Fully connected |             84 input, 43 output             |
|     Softmax     |     normalized probability distribution     |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

EPOCHS = 30
BATCH_SIZE = 128
RATE = 0.0008
KEEP_PROB = 0.7

optimizer = tf.train.AdamOptimizer(learning_rate = RATE)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of >0.99
* validation set accuracy of 0.957
* test set accuracy of  0.933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  The standard LeNet-5

* What were some problems with the initial architecture?

  Even with lots of parameter tuning, the model does not generate satifying accuracy on test data (<0.9). I found out that the model was overfitted because on the validation data it reaches over 0.97 accuracy.  

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  I added a dropout layer with a KEEP_PROB = 0.7. As introduced in the lecture, adding a dropout layer is a good mechanism to reduce a overfitted model. It turns out that the dropout layer decreases the accuracy on validation set (now ~0.95), and the accuracy on the test set is improved (now >0.93).

* Which parameters were tuned? How were they adjusted and why?

  EPOCHS = 30

  I noticed that with EPOCH=20 the model is still in its process of improving the accuracy on validation set. Therefore I increase EPOCH to 30, and it seems that the accuracy score will stay around 0.95.
  RATE = 0.0008

  The LEARN_RATE is decreased just by a little (0.0002) to slow down the learning process.

  sigma = 0.01

  Originally sigma=0.1, but I found by reducing sigma can provide much better random-generated initial weights for this specific task.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![alt text][image3]

![alt text][image6] ![alt text][image5] ![alt text][image8] ![alt text][image7] ![alt text][image4]

The fourth image might be difficult to classify because of reflection and color shifting, but the model still correctly predicts it to be "Keep-left".

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9]

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|   Turn-left-ahead    |   Turn-left-ahead    |
| Yield   | Yield 					|
| Roundabout-mandatory | Roundabout-mandatory |
| Keep-left	| Keep-left	|
|  Speed-limit-20km-h  |  Speed-limit-20km-h  |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

![alt text][image10]

The model is relatively sure about all five test images. For simplicity, the last image (Speed-limit-20-km-h) is analyzed.

For the fifth image, the model is confidently sure that this is a Speed-limit-20-km-h sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         | Speed-limit-20-km-h |
| 0.008     | Speed-limit-30-km-h |
| ...			| No passing for vehicles over 3.5 metric tons	|
| ...	      | Speed-limit-50-km-h	|
| ...				 | Speed-limit-100-km-h |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?