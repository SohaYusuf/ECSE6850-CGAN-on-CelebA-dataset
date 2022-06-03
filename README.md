# Condition-GAN-on-CelebA-dataset
We train a Conditional Generative Adversarial Network (cGAN) on CelebA dataset containing 202,599 RGB images
and one-hot attribute vectors (condition) associated with each image. In cGAN, generator takes condition y and
noise z as input and passes them through a de-convolutional layer to create corresponding embedding y’ and z’,
which are then passed through rest of generator model to output a 64x64x3 image. Discriminator takes concatenated
real and fake image I along with the condition y which is broadcasted and passed through a convolutional layer.
Image I is also passed through a convolutional layer in the discriminator to create embedding of the same dimension
32x32x64. Discriminator and generator models are trained alternatively in TensorFlow 2.6.0 with Adam optimizer
and binary cross-entropy loss function. After training, the trained generator is able to generate images according to
the given condition.
1. Generator
Generator model consist of 6 de-convolutional layers, each having ReLU activation except the last layer
which uses tanh activation function. First, noise z and label y are reshaped to 1x1x100 and 1x1x5
respectively and passed through de-convolutional layers. Deconv1 and Deconv2 creates noise embedding
z’ of dimension 4x4x512 and label embedding y’ of dimension 4x4x512. Concatenated embedding of
dimension 4x4x1024 is then passed through de-convolutional layers i.e. Deconv3-6 to output a generated
image of dimension 64x64x3. Details of each layer and output shapes are shown in the figure below:
![image](https://user-images.githubusercontent.com/102180459/171958315-119f32ca-1099-49d1-89b1-a5c3180dfd11.png)
2. Discriminator
Discriminator model consist of 5 convolutional layers and one fully-connected layer, each having Leaky
ReLU activation except the last layer which uses sigmoid activation function. Model takes concatenated
real and generated image as I along with the label y as input and outputs the probability from 0 to 1. Label
y is broadcasted to y’ of dimension 64x64x5 and passed through a convolutional layer with 64 (2,2) filters.
Image I is also passed through a convolutional layer before concatenation with y’. Concatenated embedding
of dimension 32x32x128 is then passed three convolutional layers i.e. Conv3-5. The output of Conv5 is
connected to a fully-connected layer with one node and sigmoid activation function. Details of each layer
and output shapes are shown in the figure below:
![image](https://user-images.githubusercontent.com/102180459/171958353-11717801-c806-408f-9227-02373624f387.png)
3. Dataset
The dataset contains 202,599 RGB images and attribute vectors associated with each image. Each 64x64x3
image has a corresponding attribute vector that describes the condition of the image. All images are
normalized to 0-1 by dividing each image by 255 and all attributes are converted to 5-dimensional one-hot
labels. For example, label [1,1,0,0,0] will represent a male with black hair who is not smiling, not young
and does not have an oval face. Labels are in the order: Black hair, Male, Oval Face, Smiling and Young.
Some examples from the dataset are shown in figure 7.
After loading all the data using np.load(), it has dimensions:
• Training images: 202599 × 64 × 64 × 3
• Training labels: 202599 × 5 × 1
