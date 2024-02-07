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
import losses
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
PATH_PREDICT_SAVE = os.path.join(ASSETS_DIR, "predict")
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
     aug_images = niftiSave.load_images(PATH_RAW_IMAGE, normalize=True)
     aug_masks = niftiSave.load_images(PATH_RAW_MASK, normalize=True)

     #Prepare model
     myModel = unetObj.create_unet_model(filter_num=filter_num)
     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     loss = unetObj.j_dice_coef_loss
     ## SHOULD EXPERIMENT WITH SMOOTHING VALUE IN J_IOU_LOSS
     metrics = [unetObj.j_dice_coef, unetObj.j_iou, losses.bce_dice_loss, losses.dice_loss, losses.bce_jaccard_loss, losses.jaccard_loss]
     myModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

     #Prepare callbacks
     myModelSavePath = os.path.join(DEFAULT_LOGS_DIR, f"2D_fn{filter_num}-bs{batch_size}-lr{learning_rate}.h5")
     earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1)
     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=myModelSavePath,monitor='val_loss',save_best_only=True,verbose=1,mode="min")

     #Do fit
     myModel_trained = myModel.fit(x=aug_images, y=aug_masks, validation_split=0.25, batch_size=batch_size, epochs=unetObj.epochs, shuffle=True, callbacks=[earlystopper, reduce_lr, checkpoint_callback])
     myModelHistorySavePath = os.path.join(DEFAULT_LOGS_DIR, f"2D_fn{filter_num}-bs{batch_size}-lr{learning_rate}.npy")
     np.save(myModelHistorySavePath, myModel_trained.history)

#base fn=32, bs=16, lr=0.0001
# fn 16 - 32 - 64
# bs 8 - 16 - 32
# lr 0.001 - 0.0001 - 0.00001
main_trainer(epochs=10, filter_num=32, batch_size=16, learning_rate=0.0001)

# ################################
# #||                          #||
# #||        Predictor         #||
# #||                          #||
# ################################
def predict(model_path: str):
     predictorObj = predictor.predictor(model=models.load_model(model_path, compile=False), 
                         image_array=niftiSave.load_images(PATH_AUG_IMAGE, normalize=True), 
                         mask_array=niftiSave.load_images(PATH_AUG_MASK, normalize=True))

     ran_image, ran_mask, predicted_mask = predictorObj.predict_for_3d()
     niftiSave.save_images(save_path=PATH_PREDICT_SAVE, save_prefix="r", img_iterable=ran_image, mask_bool=False)
     niftiSave.save_images(save_path=PATH_PREDICT_SAVE, save_prefix="m", img_iterable=ran_mask, mask_bool=True)
     niftiSave.save_images(save_path=PATH_PREDICT_SAVE, save_prefix="p", img_iterable=predicted_mask, mask_bool=True)

# model_path = os.path.join(DEFAULT_LOGS_DIR, "fn32-bs16-lr0.0001.h5")
# predict(model_path=model_path)