# Breast_Cancer_Deep_Learning
Deep learning and transfer learning improve breast cancer detection by training deep neural networks on large datasets, capturing intricate patterns in mammograms, ultrasounds, and histopathology slides, and leveraging pre-trained models for efficient learning.


The code then loads the breast cancer images from the "benign" and "malignant" directories using the Data_load() function and stores them in separate numpy arrays for training and testing datasets.

Next, it defines functions for plotting the data and accuracy visualization. The Plot_data() function plots a grid of images using matplotlib. 

Benign Data:
![download](https://github.com/ChakradharMudili/Breast_Cancer_Deep_Learning/assets/135093110/592ed000-acf6-45d3-a1be-ebe8ddbf5c5a)


Malignant Data:
![download](https://github.com/ChakradharMudili/Breast_Cancer_Deep_Learning/assets/135093110/40bfeffc-96e4-493f-9037-b2438723455f)

The accuracy_plot() function creates a plot to visualize the training and validation accuracy as well as the training and validation loss during the model training.

The code proceeds with building a deep learning model using the Keras library. It defines a function build_model() that constructs a sequential model with multiple convolutional and pooling layers, followed by dropout, batch normalization, and dense layers. The model is compiled with binary cross-entropy loss and Adam optimizer.

The code then applies the ImageDataGenerator to generate augmented training data and builds and trains the model using the training data. It also evaluates the model's performance on the testing data and plots the accuracy using the accuracy_plot() function.

Afterwards, the code demonstrates the use of pre-trained models such as ResNet50, DenseNet201, MobileNet, NASNetLarge, InceptionResNetV2, and NASNetMobile for breast cancer classification. It builds models by using these pre-trained models as the backbone and fine-tuning them with additional layers. The models are trained, evaluated, and their accuracy is plotted.

Finally, the code creates plots to compare the training and validation accuracy of different models.

Overall, the code showcases the implementation of deep learning models and transfer learning techniques for breast cancer classification.


