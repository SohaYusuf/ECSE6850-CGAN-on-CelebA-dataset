# In[88]:


"""
Author - SOHA YUSUF
"""


# In[89]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow import keras
from tensorflow.keras.layers import Add, Conv2D, Conv2DTranspose, Flatten, BatchNormalization, Reshape, ReLU,\
    Concatenate, Activation, Dense, LeakyReLU
from tqdm import tqdm


# In[90]:


# load training data images and labels
REAL_IMAGES = np.load('data/images.npy')           # (202599,64,64,3)
ATTRIBUTES = np.load('data/attributes5.npy')       # (202599,5)
print(REAL_IMAGES.shape)
print(ATTRIBUTES.shape)


# In[91]:


# initialize batch size
batch_size = 128

# build training dataset
with tf.device('/GPU:3'):
    train_dataset = tf.data.Dataset.from_tensor_slices((REAL_IMAGES, ATTRIBUTES))
    train_dataset = train_dataset.batch(batch_size)
    # create a map for training dataset
    train_dataset = (train_dataset.map(lambda x, y:
                                   (tf.divide((tf.cast(x, tf.float32)),255.0),tf.cast(y, tf.float32))))


# In[92]:


print(len(train_dataset))


# In[93]:


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

with tf.device('/GPU:3'):
    generator = generator_model()
    generator.summary()


# In[94]:


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

with tf.device('/GPU:3'):
    discriminator = discriminator_model()
    discriminator.summary()


# In[95]:


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


# In[96]:


# initialize conditional GAN
cond_gan = ConditionalGAN(discriminator=discriminator, generator=generator, latent_dim=100)


# In[97]:


# load saved weights
cond_gan.load_weights('cgan_weights')


# In[98]:


# generate images
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
            #plt.show()
            
    return all_fakes, fakes


# In[99]:


# compute embeddings 
def compute_embeddings(dataloader):
    image_embeddings = []

    for _ in tqdm(range(1000)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)


# In[100]:


# compute FID
def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid, mu1, mu2


# In[101]:


# compute IS
def compute_is(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1)) 
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


# In[102]:


# initialize the batch of images to generate
batches = 5
with tf.device('/GPU:3'):
    # initialize trained generator
    trained_generator = cond_gan.generator
    # generate images and plot them with labels
    all_fakes, fakes = generate_and_plot_images(train_dataset, trained_generator, batches)


# In[103]:


# resize all generated images from (64,64,3) to (299,299,3) for InceptionV3 network
with tf.device('/GPU:3'):
    resized_fakes = []
    
    for fake in all_fakes:
        resized_f = tf.image.resize(fake, (299,299))
        resized_f = tf.reshape(resized_f, (batch_size,299,299,3))
        resized_fakes.extend(resized_f)
        
    min_fake = tf.reduce_min(resized_fakes).numpy()
    max_fake = tf.reduce_max(resized_fakes).numpy()
        
    # print(len(resized_fakes))
    # print(min_fake)
    # print(max_fake)
    
    resized_fakes = (resized_fakes - min_fake)/(max_fake-min_fake)
    
    # print(tf.reduce_min(resized_fakes))
    # print(tf.reduce_max(resized_fakes))
    
    plt.imshow(resized_fakes[0])
    plt.show()


# In[104]:


# resize the real images from (64,64,3) to (299,299,3) for InceptionV3 network
with tf.device('/GPU:3'):
    resized_real = []
    for x,y in train_dataset.take(batches):           # batches for generated and real images should be equal
        resized_r = tf.image.resize(x, (299,299))
        resized_r = tf.reshape(resized_r, (batch_size,299,299,3))
        resized_real.extend(resized_r)
        
    print(len(resized_real))
    print(tf.reduce_min(resized_real))
    print(tf.reduce_max(resized_real))
    
    plt.imshow(resized_real[1])
    plt.show()


# In[105]:


with tf.device('/GPU:3'):
    # create dataset for resized generated images
    gen_dataset = tf.data.Dataset.from_tensor_slices(resized_fakes).batch(batch_size)
    gen_dataset = (gen_dataset.map(lambda x: x))
    # create dataset for resized real images
    real_dataset = tf.data.Dataset.from_tensor_slices(resized_real).batch(batch_size)
    real_dataset = (real_dataset.map(lambda x: x))
    
    print(len(gen_dataset), gen_dataset)
    print(len(real_dataset), real_dataset)


# In[106]:


# load pre-trained inceptionv3 model
with tf.device('/GPU:3'):
    inception_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_shape=(299,299,3)
)


# In[ ]:


with tf.device('/GPU:7'):
    
    # compute embeddings for real images
    real_image_embeddings = compute_embeddings(real_dataset)
    
    # compute embeddings for generated images
    generated_image_embeddings = compute_embeddings(gen_dataset)


# In[ ]:


# compute FID for all batches generated images
fid, mu_real, mu_gen = calculate_fid(real_image_embeddings, generated_image_embeddings)
# compute IS for all batches of real images
inception_score_mean , inception_score_std = compute_is(generated_image_embeddings, splits=10)

print('Fréchet Inception Distance (FID) for all generated images: ', fid)
print('Inception score (IS) for all generated images: ',inception_score_mean)


# In[ ]:


# vary label y and generate the image
with tf.device('/GPU:4'):
    # manually vary y to generate the image
    label = [0,0,0,1,0]
    y = tf.reshape(tf.cast(label,tf.float32),(1,5,1))   # label
    z = tf.random.normal(shape=(1,100,1))               # noise vector
    my_image = (trained_generator.predict([z,y]))        # generate image
    print(my_image.shape)
    plt.imshow(my_image[0])
    plt.title(label)
    plt.show()


# In[ ]:


resized_my_image = tf.reshape(tf.image.resize(my_image, (299,299)), (1,299,299,3))
my_embedding = inception_model.predict(resized_my_image)

my_is, _ = compute_is(my_embedding, splits=10)
my_fid, my_mu_real, my_mu_gen = calculate_fid(real_image_embeddings, my_embedding)

print('My FID: ', my_fid)
print('My IS: ', my_is)
