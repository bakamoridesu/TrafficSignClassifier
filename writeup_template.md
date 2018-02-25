# **Traffic Sign Recognition** 

## Self Driving Car Nanodegree. Project 2.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

After loading the data I calculated the size of training and testing data. I also wanted to know the shape of loaded images and the number of classes in the dataset.

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

Then I wanted to take a look at the images:

![samples](/examples/samples.jpg)

What I actually did in the project is I plotted one random image for every class in the dataset. This visualization showed that not all the signs are nicely centered. Some of the images are rotated, some of them are zoomed and some are sheared. A lot of pictures were taken at nighttime, so there is probably no need of keeping 3 color channels. 

Then I wanted to know the number of sample images representing each class. The pyplot bar is ideal choice for this kind of visualization:

![first plot](/examples/first_plot.jpg)

This step is very useful for future data augmentation. There is a large white space in this bar plot. It means that some classes are represented by much larger set of samples than the other ones. To normalize this situation I don't want to augment full dataset, I just want to fill up most of the white space of this bar.

After those few steps I knew how to pre-process image and how to augment the data for better results.

#### 1. Data pre-processing and augmentation.
Before pre-processing data I decided to generate more images for those classes which lack of samples. 
The `keras.preprocessing.image.ImageDataGenerator` class was chosen for data augmentation.
The idea is to generate more samples for the classes having the less samples:
- 5x more sample batches for the classes having lesser than 400 samples
- 3x more sample batches for the classes having lesser than 700 samples
- 1 more sample batch for the classes having lesser than 1500 samples
After generating new data (which is really just a little bit rotated, shifted, sheared and zoomed copies of existing images in the dataset) the Sample-Per-Class ratio become more balanced:

![before_after](/examples/before_after.jpg)


The first pre-processing step is converting image to grayscale. 
Then I normalized image by dividing every pixel by 255.
Finally I wanted to remove most of the noize and (if possible) to 'highlight' meaningful part of the image. I tried different approaches and finally I decided to use `exposure` from `scikit-learn`. The methods `equalize_adapthist` and `adjust_sigmoid` are doing almost exactly what I wanted to see. 
The next few pairs of images show the difference between initial and pre-processed data.

![preprocess](/examples/preprocess.jpg)

After these steps the data was ready to be passed into the deep learning model.

#### 2. Model architecture.

My final model consisted of 3 convolutional layers and 3 fully-connected layers. Every layer contains relu and dropout, after every convolutional layer goes pooling layer. The output layer has 43 outputs. I also used L2 regularization for model weights.

![preprocess](/examples/model.jpg)

#### 3. Training process.

The data preprocessing and augmentation steps resulted really good training data, and for the first try I earned 96% of test accuracy. Then I spent a lot of time improving this result by fine-tuning the parameters.
The key parameters were:
- `beta` for L2 regularization. **Final value is 0.0003**
- dropout values for convolutional layers **Final values are keep_prob = .9 for first two layers and .8 for the last layer**

#### 4. Model results:

Final **Training accuracy** : **99.6%**
Final **Validation accuracy** : **99.1%**
Final **Validation accuracy** : **97.6%**

The model has a little bit of overfitting, probably fine-tuning the dropout values is the way to prevent it.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![bicycles_crossing][new_images/bicycles_crossing.jpg] 
![no_entry][new_images/no_entry.jpg] 
![road_works][new_images/road_works.jpg] 
![roundabout][new_images/roundabout.jpg] 
![turn_right][new_images/turn_right.jpg] 


