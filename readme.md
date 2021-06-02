# IMAGE COMPRESSION

This repository hosts code, theroy, observations and results for my experiments with the image compression techniques listed below:
1. Autoencoder
2. Singular Value Decomposition
3. GAN

***
## Autoencoder:
An autoencoder is trained to find an encoding and decoding algorithm simultaneously for a given set of inputs. It results in two models which in series, poses as an identity function. Seperately, they function as each other's opposites. An image encoded using the first half can only be decoded with the second half, so they are called encoder and decoder respectively.

Here we train an autoencoder on a dataset of images of handwritten numeric digits. The resulting model is used interactively through an app made with Flask. 

## Singular Value Decomposition:
This mathematical method is used to expres any higher ranking matrix as a sum of dot product of rank-1 matrices. 

<img src="./images/SVD.png" alt="svd" width="300"/>

Here we use the first k elements in the singular value decomposition of our image and analyse how much compression we can get and the factors that affect it.

<img src="./images/svddemo.jpg" alt="svd" width="500"/>