# Image-Super-Resolution-Based-on-Generative-Adversarial-Networks

Dataset DIV2K & Process the data from Different Resolution to 32 x 32 Low Resolution image. 

```python
train_dir = "/content/drive/MyDrive/Experiment/srgan_dataset/train_img" 
def data_preprocess(data):
    for img in os.listdir( train_dir + "/" + data):
        img_array = cv2.imread(train_dir + "/" + data + "/" + img)
        
        img_array = cv2.resize(img_array, (128, 128))
        lr_img_array = cv2.resize(img_array,(32, 32))
        cv2.imwrite(train_dir + "/HR_images/" + img, img_array)
        cv2.imwrite(train_dir + "/LR_images/"+ img, lr_img_array)
data_preprocess( "DIV2K_train_HR") 

```

Generator & Discriminator Model 

```python
#Define blocks to build the generator
def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])

def upscale_block(ip):
    
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

#Generator model
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)

#Descriminator block that will be used to construct the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    
    return disc_model


#Descriminartor model
def create_disc(disc_ip):

    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)


def build_vgg(hr_shape):
    
    vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
    
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

#Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    
    gen_features = vgg(gen_img) # Content Loss
    
    disc_model.trainable = False
    validity = disc_model(gen_img) # Adversarial Loss
    
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features]) 

```
Read image data & convert to Matrix & also append value into a list also normalization. 
```python 

lr_list = os.listdir("/content/drive/MyDrive/Experiment/srgan_dataset/train_img/LR_images")

lr_images = []
for img in lr_list:
    img_lr = cv2.imread("/content/drive/MyDrive/Experiment/srgan_dataset/train_img/LR_images/" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)   


hr_list = os.listdir("/content/drive/MyDrive/Experiment/srgan_dataset/train_img/HR_images")
   
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("/content/drive/MyDrive/Experiment/srgan_dataset/train_img/HR_images/" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)   

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

#Scale values
lr_images = lr_images / 255.0
hr_images = hr_images / 255.0


lr_train = lr_images
hr_train = hr_images

hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_gen(lr_ip, num_res_block = 16)

discriminator = create_disc(hr_ip)

discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

vgg = build_vgg((128,128,3))
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)


gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer= keras.optimizers.Adam(learning_rate=0.001))
batch_size = 10
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
```
Train & Save the Model.
```python
#Enumerate training over epochs
epochs = 220
gan_loss = []
discr_loss = []
ep = []
for e in range(epochs):
    
    fake_label = np.zeros((batch_size, 1)) # Assign a label of 0 to all fake (generated images)
    real_label = np.ones((batch_size,1)) # Assign a label of 1 to all real images.
    
    #Create empty lists to populate gen and disc losses. 
    g_losses = []
    d_losses = []
    
    #Enumerate training over batches. 
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b] #Fetch a batch of LR images for training
        hr_imgs = train_hr_batches[b] #Fetch a batch of HR images for training
        
        fake_imgs = generator.predict_on_batch(lr_imgs) #Fake images
        
        #First, train the discriminator on fake and real HR images. 
        discriminator.trainable = True
        d_loss_gen, acc_fack = discriminator.train_on_batch(fake_imgs, fake_label)

        d_loss_real, acc_real = discriminator.train_on_batch(hr_imgs, real_label)
        #Now, train the generator by fixing discriminator as non-trainable
        discriminator.trainable = False
        
        #Average the discriminator loss, just for reporting purposes. 
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        
        #Extract VGG features, to be used towards calculating loss
        image_features = vgg.predict(hr_imgs)
     
        #Train the generator via GAN. 
        #Remember that we have 2 losses, adversarial loss and content (VGG) loss
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
        
        #Save losses to a list so we can average and report. 
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    print(f"Acc Fack: {acc_fack*100}% & Acc Real: {acc_real*100}%")
    #Convert the list of losses to an array to make it easy to average    
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
    #Calculate the average losses for generator and discriminator
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    gan_loss.append(g_loss)
    discr_loss.append(d_loss)
    ep.append(e)
    print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)
    generator.save("/content/drive/MyDrive/Experiment/ganlr99.h5")
``` 

Load the model & Test the model from different low resolution image to Super Resolution image and calculate the PSNR & SSIM. 

```python
#Test - perform super resolution using saved generator model
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
def compute_psnr(original_image, generated_image):
    
    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)
    
    psnr = tf.image.psnr(original_image, generated_image, max_val = 1.0)
    
    return tf.math.reduce_mean(psnr, axis = None, keepdims = False, name = None)

def compute_ssim(original_image, generated_image):
    
    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)
    
    ssim = tf.image.ssim(original_image, generated_image, max_val = 1.0, filter_size = 11, filter_sigma = 1.5, k1 = 0.01, )
    
    return tf.math.reduce_mean(ssim, axis = None, keepdims = False, name = None)

#generator = load_model('gano200.h5', compile=False)
generator = load_model('/content/drive/MyDrive/Experiment/ganlr99.h5', compile=False)
img_array = cv2.imread("c.jpg")

img_array = cv2.resize(img_array, (128, 128))
LR_img_array = cv2.resize(img_array,(32, 32))
#--------------------------------------------------------------------#
#sreeni_lr = cv2.imread("/content/0896.png")
#sreeni_hr = cv2.imread("/content/u.png")

sreeni_lr = LR_img_array
sreeni_hr = img_array



sreeni_lr = cv2.cvtColor(sreeni_lr, cv2.COLOR_BGR2RGB)
sreeni_hr = cv2.cvtColor(sreeni_hr, cv2.COLOR_BGR2RGB)

sreeni_lr = sreeni_lr / 255.
sreeni_hr = sreeni_hr / 255.

sreeni_lr = np.expand_dims(sreeni_lr, axis=0)
sreeni_hr = np.expand_dims(sreeni_hr, axis=0)

generated_sreeni_hr = generator.predict(sreeni_lr)
psnr = "{:.3f}".format(compute_psnr(sreeni_hr, generated_sreeni_hr))
ssim = "{:.3f}".format(compute_ssim(sreeni_hr, generated_sreeni_hr))
print(f"PSNR - Peak to Signal Noise Ratio: {psnr}")

print(f"SSIM - Structural Similarity Index: {ssim}")
plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.title('LR Image')
plt.imshow(sreeni_lr[0,:,:,:])
plt.subplot(132)
plt.title('SR image PSNR\SSIM: ({}\{})'.format(psnr, ssim))
plt.imshow(generated_sreeni_hr[0,:,:,:])
#plt.savefig('SR.png')

plt.subplot(133)
plt.title('Orig. HR image')
plt.imshow(sreeni_hr[0,:,:,:])
plt.savefig('result.png')

plt.show()
```


Output: 
![alt text](https://drive.google.com/file/d/1pftJUAoq6lHuHU3F-Gg0vMB5NXv4aly5/view?usp=drivesdk) 





