# In[1]:


"""
Author - SOHA YUSUF
"""


# In[2]:


import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, Reshape, ReLU,     Concatenate, Activation, Dense, LeakyReLU


# In[3]:


# load training images and labels
REAL_IMAGES = np.load('data/images.npy')              # (202599,64,64,3)
ATTRIBUTES = np.load('data/attributes5.npy')          # (202599,5)


# In[4]:


print(REAL_IMAGES.shape)
print(ATTRIBUTES.shape)


# In[5]:


# function for writing the attributes of image
def attribute(att):
    if att[0]==1:
        print("Black Hair")
    elif att[0]==0:
        print("Not Black Hair")
    if att[1]==1:
        print("Male")
    elif att[1]==0:
        print("Not Male")
    if att[2]==1:
        print("Oval Face")
    elif att[2]==0:
        print("Not Oval Face")
    if att[3]==1:
        print("Smiling")
    elif att[3]==0:
        print("Not Smiling")
    if att[4]==1:
        print("Young")
    elif att[4]==0:
        print("Not Young")


# In[6]:


# visualize the images with labels 
print('Shape of images: ', REAL_IMAGES.shape)
i = np.random.randint(0, 202599, size=1)[0]
att = ATTRIBUTES[i]
print('Training sample out of 202599: ',i)
print(attribute(att))
plt.imshow(REAL_IMAGES[i])
plt.title(att)


# In[7]:


# initialize the hyper parameters
batch_size = 128            # batch size
EPOCHS = 20                 # total epochs
learning_rate = 0.0001      # learning rate
momentum = 0.5              # momentum term for adam optimizer

# initialize the dimensions
num_channels = 3            # number of channels for images
num_classes = 5             # number of attributes 
image_size = 64             # size of each image
latent_dim = 100            # dimension of noise vector


# In[12]:


with tf.device('/GPU:2'):
    # build training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((REAL_IMAGES, ATTRIBUTES))
    # take batches of training dataset
    train_dataset = train_dataset.batch(batch_size)
    # shuffle the training dataset
    #train_dataset = train_dataset.shuffle(buffer_size=500)
    # create a map for training dataset
    train_dataset = (train_dataset.map(lambda x, y:
                                   (tf.divide(tf.cast(x, tf.float32), 255.0),tf.cast(y, tf.float32))))


# In[13]:


print(len(train_dataset))


# In[14]:


# view if data and attributes are loaded correctly
for X_train, Y_train in train_dataset.take(1):
    print('Shape of images: ', X_train.shape)
    i = np.random.randint(0, 9, size=1)[0]
    att = ATTRIBUTES[i]
    print('Training sample out of 202599: ',i)
    print(attribute(att))
    plt.imshow(X_train[i])
    plt.title(ATTRIBUTES[i])


# In[16]:


# Create the generator.
def generator_model():
    
    # create embedding of noise z
    z_shape = (100,1)
    z = tf.keras.Input(shape=z_shape)
    # (100,1)
    R1 = Reshape((1,1,100))(z)
    # (1,1,100)
    Deconv1 = Conv2DTranspose(512, (4,4), strides=(1, 1), padding='valid')(R1)
    B1 = BatchNormalization()(Deconv1)
    A1 = ReLU()(B1)  #(4,4,512)
    
    # create embedding for condition y
    y_shape = (5,1)
    y = tf.keras.Input(shape=y_shape)
    # (5,1)
    R2 = Reshape((1,1,5))(y)
    # (1,1,5)
    Deconv2 = Conv2DTranspose(512, (4,4), strides=(1, 1), padding='valid')(R2)
    B2 = BatchNormalization()(Deconv2)
    A2 = ReLU()(B2)  #(4,4,512)
    
    # concatenate the embeddings of z and y
    Concat1 = Concatenate()([A1,A2])  # (4,4,1024)
    
    # Deconv3
    # (4,4,1024)
    Deconv3 = Conv2DTranspose(512, (5,5), strides=(2,2), padding='same')(Concat1)
    B3 = BatchNormalization()(Deconv3)
    A3 = ReLU()(B3)    # (8,8,512)
    
    # Deconv4
    # (8,8,512)
    Deconv4 = Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(A3)
    B4 = BatchNormalization()(Deconv4)
    A4 = ReLU()(B4)    # (16,16,256)
    
    # Deconv5
    # (16,16,256)
    Deconv5 = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(A4)
    B5 = BatchNormalization()(Deconv5)
    A5 = ReLU()(B5)    # (32,32,128)
    
    # Deconv6
    # (32,32,128)
    Deconv6 = Conv2DTranspose(3, (5,5), strides=(2,2), padding='same')(A5)
    B6 = BatchNormalization()(Deconv6)
    A6 = Activation('tanh')(B6)    # (64,64,3)
    
    model = tf.keras.Model(inputs=[z,y], outputs=A6, name='generator')
    
    return model

with tf.device('/GPU:2'):
    # define generator model
    generator = generator_model()
    generator.summary()


# In[17]:


# Create the discriminator.
def discriminator_model():
    
    # create embedding of input image I
    I_shape = (64,64,3)
    I = tf.keras.Input(shape=I_shape)
    # (64,64,3)
    Conv1 = Conv2D(64, (2,2), strides=(2, 2), padding='valid')(I)
    B1 = BatchNormalization()(Conv1)
    A1 = LeakyReLU(alpha=0.2)(B1)   #  (32,32,64)
    
    # create embedding of condition y
    y_shape = (5,1)
    ad = tf.zeros((1,64,64,5), dtype=tf.dtypes.float32)
    y = tf.keras.Input(shape=y_shape)
    
    # Braodcast y from (1,5) to shape (64,64,5)
    R2 = Reshape((1,5))(y)
    BR2 = Add()([ad,R2])    # (64,64,5)
    
    # Conv2
    # (64,64,5)
    Conv2 = Conv2D(64, (2,2), strides=(2,2), padding='valid')(BR2)
    B2 = BatchNormalization()(Conv2)
    A2 = LeakyReLU(alpha=0.2)(B2)  #(32,32,64)
    
    # concatenate the embeddings of z and y
    Concat1 = Concatenate()([A1,A2])  # (32,32,128)
    
    # Conv3
    # (32,32,128)
    Conv3 = Conv2D(256, (2,2), strides=(2,2), padding='valid')(Concat1)
    B3 = BatchNormalization()(Conv3)
    A3 = LeakyReLU(alpha=0.2)(B3)     # (16,16,256)
    
    # Conv4
    # (16,16,256)
    Conv4 = Conv2D(512, (2,2), strides=(2,2), padding='valid')(A3)
    B4 = BatchNormalization()(Conv4)
    A4 = LeakyReLU(alpha=0.2)(B4)    # (8,8,512)
    
    # Conv5
    # (8,8,512)
    Conv5 = Conv2D(1024, (2,2), strides=(2,2), padding='valid')(A4)
    B5 = BatchNormalization()(Conv5)
    A5 = LeakyReLU(alpha=0.2)(B5)      # (4,4,1024)
    
    # Fully connected layer
    # (4,4,1024)
    F1 = Flatten()(A5)
    D1 = Dense(1)(F1)
    A6 = Activation('sigmoid')(D1)
    
    model = tf.keras.Model(inputs=[I,y], outputs=A6, name='discriminator')
    
    return model

with tf.device('/GPU:2'):
    # define discriminator model
    discriminator = discriminator_model()
    discriminator.summary()


# In[18]:


# create callbacks for visualising images during training
class ConditionalGANmonitor(keras.callbacks.Callback):
    def __init__(self, labels, num_img=3, latent_dim=100):
        self.labels = labels
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        combined = tf.concat([random_latent_vectors,self.labels], axis=1)
        generated_images = self.model.generator([random_latent_vectors,self.labels])
        generated_images *= 255
        generated_images.numpy()
        
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            plt.imshow(img)
            plt.show()


# In[19]:


# create class for conditional GAN 
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data
        
        use_one_hot_labels = one_hot_labels[:,:,None]
        
        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator([random_latent_vectors,one_hot_labels])

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_and_real_image = tf.concat([generated_images, real_images], axis=0)
        fake_and_real_labels = tf.concat([use_one_hot_labels, use_one_hot_labels], axis=0)

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator([fake_and_real_image,fake_and_real_labels])
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator([random_latent_vectors, one_hot_labels])
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator([fake_images, use_one_hot_labels])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


# In[21]:


# initialize conditional GAN model
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
# initialize optimizer and loss function for conditional GAN
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate,momentum),
    g_optimizer=keras.optimizers.Adam(learning_rate,momentum),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)
# callbacks for plotting generated images
monitor = ConditionalGANmonitor(labels=ATTRIBUTES[3:6], num_img=3, latent_dim=100)

with tf.device('/GPU:2'):
    # initialize start time
    start_time = time.time()
    # train the model
    history = cond_gan.fit(train_dataset, epochs=EPOCHS, callbacks=[monitor])
    # initialize end time
    end_time = time.time()
    # print total runtime
    print("Time took: {:4.2f} min".format((end_time - start_time)/60))


# In[25]:


g_loss = history.history['g_loss']        # generator loss
d_loss = history.history['d_loss']        # discriminator loss

# plot the loss curves
plt.figure(1)
plt.figure(figsize=(8, 8))
plt.plot(g_loss, label=f'Generator Loss')
plt.plot(d_loss, label=f'Discriminator Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title(f'Loss Curve ({batch_size} batch size)')
plt.grid()
plt.xlabel('# epochs')
plt.show()


# In[26]:


# plot the generated images with labels
def generate_and_plot_images(dataset, model, batches):
    
    all_fakes = []

    for x,y in dataset.take(batches):
        
        m = x.shape[0]
        y = tf.reshape(y, (m,5))
        z = tf.random.normal(shape=(m, 100))
        model_input = tf.concat([z,y], axis=1)
        fakes = model.predict([z,y])
        all_fakes.append(fakes)

        for i, img in enumerate(fakes):
            print(y[i].numpy())
            img = tf.keras.preprocessing.image.array_to_img(img)
            plt.imshow(img)
            plt.show()
            
    return all_fakes


# In[28]:


with tf.device('/GPU:1'):
    # initialize trained generator
    trained_generator = cond_gan.generator
    # generate images with labels
    generate_fakes = generate_and_plot_images(train_dataset, trained_generator, batches=1)


# In[29]:


# save weights for the model
cond_gan.save_weights('cgan_weights')
