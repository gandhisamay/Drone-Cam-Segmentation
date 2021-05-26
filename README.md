# Semantic Segmenatation
Drone Cam Images from bird view are taken and segmented into 23 different classes. The Kaggle dataset can be downloaded from the this [website](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)

# Dataset
The dataset contains `400 images` which are of dimensions ` (3,4000,6000)`. Out of which `390 images` have been taken for training and remaining 10 haven been use for testing the model. The dataset contains the following, 
1) Images from the Cam
2) Label Images(RGB type)
3) Colored Masks Images(RGB type)
4) CSV file containing color channels values for RGB images

For training the model within available hardware resources, the images are resized to a size of `(3,256,256)`.

# Model
The `U-net model` has been implemented using Pytorch with some minor changes for producing best results. A `BatchNorm layer` has been introduced in the model for faster training which was not present in the original model since it was discovered a year earlier before BatchNorm. 

Also, the number of channels have been reduced to curb with memory issues. The number of channels were reduced to `one-fourth` of the optimal values obtained in U-net architecture.
![network architecture](https://i.imgur.com/jeDVpqF.png)

# Loss Function
`Dice Loss` function has been used for training the model. The dice loss function along with `one-hot encoding` rather than using `torch.argmax()` function since the argmax function is not differentiable.

## Notes on Memory
Google Colab's Tesla T4 GPU has been used to train the model. 

## License
Credit : Just drop a star as a credit if you use any part of the implementation

