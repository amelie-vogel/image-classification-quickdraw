# Image Classification using CNN - Google Quick, Draw!
Image recognition and classification based on Convolutional Neural Networks to identify up to 8 classes of animals.

## Dataset

### Description

The complete dataset comes from the [Google Quick, draw! Dataset](https://quickdraw.withgoogle.com/data/). It is a collection of 50 million drawings across 345 categories, contributed by players of the game Quick, Draw!. The drawings were captured as timestamped vectors, tagged with metadata including what the player was asked to draw and in which country the player was located.

### Overview

For the purpose of the Image Recognition project, I decided to focus on animals, and selected up to 8 classes. I extracted 10,000 images of each class, using the simplified drawing files that had been made available (.ndjson) on the [Google Cloud Platform.](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified;tab=objects?prefix=/). These simplified files are vectors without the timing information, positioned and scaled into a 256x256 region, that I resized into 28x28 (std. MNIST dataset sizes). 

![image](https://user-images.githubusercontent.com/63364114/91064433-88ba8d00-e62f-11ea-9374-c99ff645a90d.png)

![image](https://user-images.githubusercontent.com/63364114/91064469-94a64f00-e62f-11ea-92b3-a52d6d5e8f46.png)

![image](https://user-images.githubusercontent.com/63364114/91064485-996b0300-e62f-11ea-8bcc-a1480d5f0b66.png)

![image](https://user-images.githubusercontent.com/63364114/91064503-9e2fb700-e62f-11ea-9b04-7be9870800ee.png)

## Architecture of the CNN

#### The model is a 10-layer CNN which is composed of:
- A convolution layer with patches of size 5x5
- A Max pooling layer
- A convolution layer with patches of size 3x3
- A Max pooling layer of size 2x2
- A dropout layer with a rate of 40%
- A flatten layer
- A fully connected layer and rectifier activation function
- A fully connected layer and rectifier activation function
- An output layer (fully connected layer) with 8 classes and softmax activation function

##### Code block:
```
    convnet = models.Sequential()
    convnet.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(small_side, small_side, 1)))
    convnet.add(layers.MaxPooling2D((2, 2)))
    convnet.add(layers.Conv2D(128, (3, 3), activation='relu'))
    convnet.add(layers.MaxPooling2D((2, 2)))
    convnet.add(layers.Dropout(0.4))
    convnet.add(layers.Flatten())
    convnet.add(layers.Dense(128, activation='relu'))
    convnet.add(layers.Dense(50, activation='relu'))
    convnet.add(layers.Dense(nb_classes, activation='softmax'))
```

## Results and Learning Curves

After training with a batch size of 128 and using 10 to 15 epochs for training both the training and test set of images:

### Classification accuracy for 2 classes (20'000 training examples):
Convolutional Neural Network: 95.83 %

![image](https://user-images.githubusercontent.com/63364114/91077095-c7a50e80-e640-11ea-88db-75332396c6ae.png)


### Classification accuracy for 4 classes (40'000 training examples):
Convolutional Neural Network: 87.83 %

![image](https://user-images.githubusercontent.com/63364114/91078587-24a1c400-e643-11ea-92d0-0381356b3fe6.png)


### Classification accuracy for 8 classes (80'000 training examples):
Convolutional Neural Network: 88.45 %

![image](https://user-images.githubusercontent.com/63364114/91168170-77c45700-e6d5-11ea-931a-c90e2f7c5a4f.png)

## Confusion Matrix

![image](https://user-images.githubusercontent.com/63364114/91332537-9f4a1b00-e7cc-11ea-9ff1-c8e852e1fb1a.png)

## Conclusion

The relatively low results of the model (especially for the 8 classes) can be explained by three main factors:

1) The quality bias: The quality of the drawings in the dataset is heterogeneous and the labels are sometimes unrecognizable. Drawing animals is indeed a complex task that requires a sense of the right proportions to make them distinguishable. 

2) The species bias: Most classes of animals have similarities, which makes the distinction difficult: a body shape topped with a head, and 2 to 4 legs for most of them.

3) The cultural bias: All people around the globe, from different cultures, don't draw animals with the same lines or shapes. For example, cows don't necessarily have patches in all countries, but giraffes do, which can lead to errors of identification.

## Contact me
[LinkedIn](https://linkedin.com/in/amelie-vogel-/)

[GitHub](https://https://github.com/amelie-vogel/)
