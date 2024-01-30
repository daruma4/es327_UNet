#Public
import json
import os
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
import numpy as np
import albumentations

#Local
import niftiSave
import unet
import predictor
import augment

#Current working directory of the project
ROOT_DIR = os.path.abspath("")
# Path to assets
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
# Path to store trained models
DEFAULT_LOGS_DIR = os.path.join(ASSETS_DIR, "trained_models")
# Path to images etc. for model
DATASET_DIR = os.path.join(ASSETS_DIR, "model_data")

PATH_RAW_IMAGE = os.path.join(DATASET_DIR, "raw\\image")
PATH_RAW_MASK = os.path.join(DATASET_DIR, "raw\\mask")
PATH_AUG_IMAGE = os.path.join(DATASET_DIR, "augmented\\image")
PATH_AUG_MASK = os.path.join(DATASET_DIR, "augmented\\mask")
PATH_NIFTI = os.path.join(DATASET_DIR, "nifti")
PATH_NIFTI_META = os.path.join(PATH_NIFTI, "meta.json")
################################
#||                          #||
#||       Nifti to PNG       #||
#||                          #||
################################
def main_nifti():
     #Load Nifti metadata and create niftisaver instance
     try:
          metadata = json.load(open(PATH_NIFTI_META))
     except FileNotFoundError:
          print(f"[ERROR] Metadata file not found at {PATH_NIFTI_META}")
     nifti_folder_list = [os.path.join(PATH_NIFTI, each) for each in os.listdir(PATH_NIFTI) if each.endswith(".nii")]

     #Open each Nifti and save images + masks
     for nifti_path in nifti_folder_list:
          nib_nifti = nib.load(nifti_path)
          nifti_file_name = os.path.basename(nifti_path)

          raw_data = nib_nifti.get_fdata()[:,:,:,0,1] # all images
          data_iterable = niftiSave.image_array_to_iterable(raw_data) # make data iterable
          cut_ranges = metadata[nifti_file_name]["ranges"]
          data_iterable_ranged = niftiSave.range_crop(cut_ranges, data_iterable) # cuts to only range specified
          data_iterable_cropped = niftiSave.crop_images_iterable(data_iterable_ranged) # crop images

          mask_bool = metadata[nifti_file_name]["mask_bool"]
          save_path = PATH_RAW_IMAGE
          if mask_bool is True:
               save_path = PATH_RAW_MASK

          
          niftiSave.save_images(save_path, metadata[nifti_file_name]["save_prefix"], data_iterable_cropped, mask_bool=mask_bool)

################################
#||                          #||
#||        Augmentor         #||
#||                          #||
################################
def main_augmentation():
     AUGMENTATIONS_LIST = albumentations.Compose(
          [
               albumentations.Blur(blur_limit=15, p=0.5),
               albumentations.HorizontalFlip(p=0.5),
               albumentations.VerticalFlip(p=0.5),
               albumentations.RandomRotate90(p=0.5)
          ]
     )

     image_array=niftiSave.load_images(PATH_RAW_IMAGE)
     mask_array=niftiSave.load_images(PATH_RAW_MASK)


     aug_raw, aug_mask = augment.do_albumentations(transform=AUGMENTATIONS_LIST, img_list=image_array, mask_list=mask_array)
     niftiSave.save_images(save_path=PATH_AUG_IMAGE, save_prefix="r", img_iterable=aug_raw, mask_bool=False)
     niftiSave.save_images(save_path=PATH_AUG_MASK, save_prefix="m", img_iterable=aug_mask, mask_bool=True)

################################
#||                          #||
#||         Trainer          #||
#||                          #||
################################
def main_trainer(img_height=256, img_width=256, img_channels=1, epochs=100, filter_num=32, batch_size=16, learning_rate=0.0001):
     #Should setup to change filter_num, batch_size and learning_rate
     unetObj = unet.unet_model(filter_num=filter_num, img_height=img_height, img_width=img_width, img_channels=img_channels, epochs=epochs)
     aug_images = niftiSave.load_images(PATH_AUG_IMAGE, normalize=True)
     aug_masks = niftiSave.load_images(PATH_AUG_MASK, normalize=True)

     #Prepare model
     myModel = unetObj.create_unet_model(filter_num=filter_num)
     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     loss = unetObj.j_iou_loss
     ## Investigation needed - why does it train on J_dice_coef_loss and not g...
     metrics = [unetObj.g_dice_coef_loss, unetObj.j_dice_coef_loss, unetObj.g_iou, unetObj.j_iou, unetObj.g_iou_loss]
     myModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

     #Prepare callbacks
     earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1)


     #Do fit
     myModel_trained = myModel.fit(x=aug_images, y=aug_masks, validation_split=0.25, batch_size=batch_size, epochs=unetObj.epochs, shuffle=True, callbacks=[earlystopper, reduce_lr])
     myModelSavePath = os.path.join(DEFAULT_LOGS_DIR, f"fn{filter_num}-bs{batch_size}-lr{learning_rate}.h5")
     myModelHistorySavePath = os.path.join(DEFAULT_LOGS_DIR, f"fn{filter_num}-bs{batch_size}-lr{learning_rate}.npy")
     myModel.save(myModelSavePath)
     np.save(myModelHistorySavePath, myModel_trained.history)

def training_routine():
     filter_nums = [16, 32, 64]
     batch_sizes = [16, 32, 64]
     learing_rates = [0.001, 0.0001, 0.00001]
     for filter_num in filter_nums:
          main_trainer(filter_num=filter_num)
     for batch_size in batch_sizes:
          main_trainer(batch_size=batch_size)
     for lr in learing_rates:
          main_trainer(learning_rate=lr)

# ################################
# #||                          #||
# #||        Predictor         #||
# #||                          #||
# ################################
def predict(model_path: str):
     predictorObj = predictor.predictor(model=models.load_model(model_path, compile=False), 
                         image_array=niftiSave.load_images(PATH_AUG_IMAGE, normalize=True), 
                         mask_array=niftiSave.load_images(PATH_AUG_MASK, normalize=True))

     ran_image, ran_mask, predicted_mask = predictorObj.random_predict()
     fig, subplots = plt.subplots(3, 1)
     subplots[0].imshow(ran_image, cmap='gray')
     subplots[0].set_title(f"Image")
     subplots[1].imshow(ran_mask, cmap='gray')
     subplots[1].set_title(f"Mask")
     subplots[2].imshow(np.reshape(predicted_mask, (256,256,1)), cmap='gray')
     subplots[2].set_title(f"Predicted Mask")
     for subplot in subplots:
          subplot.set_xticks([])
          subplot.set_yticks([])
     plt.show()

# predict(os.path.join(DEFAULT_LOGS_DIR, "fn32-bs16-lr0.0001.h5"))