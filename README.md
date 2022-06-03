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
a) Training images: 202599 × 64 × 64 × 3
b) Training labels: 202599 × 5 × 1

4. Experimental settings

Conditional GAN is implemented in TensorFlow 2.6.0, CUDA 10.2, Python 3 and trained on GPU Tesla
V100-SXM2-32GB for 84.7 minutes. Implementation is inspired by [6]. The model is trained on CelebA
dataset using Adam optimizer (momentum = 0.5), learning rate of 0.0001 and batch size of 128 for 20
epochs. Generator and discriminator models are trained by optimizing binary cross-entropy loss function.
Before training, all training images are normalized to [0,1] by dividing each image by 255.0 and all training
labels are converted to one-hot labels of type float32. Custom class of conditional GAN is initialized under
’Model’ class in keras and images are generated during training using callbacks method. To optimize the
training process, multiple GPUs are used for training and evaluation using tf.device().

5. Generated images

![image](https://user-images.githubusercontent.com/102180459/171958557-0f62892e-55ca-41ed-ac45-09d485557c8b.png)

6. Loss versus epochs

Generator loss decreases and converges are around 20 epochs from loss value of 3.8 to 0.9. Discriminator
loss increases in the start and then starts to decrease, converging at around 20 epochs. 

![image](https://user-images.githubusercontent.com/102180459/171958693-dc34feaf-f971-4c52-bcc8-29b7c7095e53.png)

7. Conclusion

We trained a deep convolutional conditional GAN by alternatively training generator and discriminator
models. We observe that generator and discriminator loss is unstable at the start but start to converge at 20
epochs. Training the model at higher batch size is computationally expensive but increase in batch size
reduces the stochasticity in the loss curves. Also, using Adam optimizer with learning rate 0.0001 and 0.5
momentum gives optimum results. FID is lower than 15 and IS is higher than 2 when batch size of 128
is used along with leaky ReLU activation function for discriminator. Finally, conditional GAN is able to
generate images for specific conditions. cGANs can be used in many applications for example, for data
augmentation in medical fields.

