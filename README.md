# Image-Super-Resolution-Based-on-Generative-Adversarial-Networks 
1. Dataset DIV2K & Process the data from High Resolution to 32 x 32 Low Resolution image. 
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
