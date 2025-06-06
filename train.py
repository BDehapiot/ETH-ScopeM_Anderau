#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models.annotate import Annotate
from bdtools.models.unet import UNet

# functions
from functions import get_tif_metadata, get_tif_data

#%% Inputs --------------------------------------------------------------------

# Path
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")
train_path = Path("data", "train")

# Procedure
annotate = 0
train = 0
predict = 1

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 0
load_name0 = ""

# preprocess
patch_size = 256
patch_overlap = 128
img_norm = "none"
msk_type = "edt"

# augment
iterations = 4000
invert_p = 0.0
gamma_p = 0.5
gblur_p = 0.5
noise_p = 0.5 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 8
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

# predict
idx = 10
load_name1 = "model_256_edt_4000-936_1"

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%% Annotate ------------------------------------------------------------------
    
    if annotate:
        
        Annotate(train_path)
    
#%% Train ---------------------------------------------------------------------
    
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(train_path.rglob("*.tif")):
            if "mask" in path.name:
                if Path(str(path).replace("_mask", "")).exists():
                    msks.append(io.imread(path))   
                    imgs.append(io.imread(str(path).replace("_mask", "")))
        imgs = np.stack(imgs)
        msks = np.stack(msks)

        unet = UNet(
            save_name="",
            load_name=load_name0,
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        # Train
        unet.train(
            
            imgs, msks, 
            X_val=None, y_val=None,
            preview=preview,
            
            # Preprocess
            img_norm=img_norm, 
            msk_type=msk_type, 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscale_factor, 
            
            # Augment
            iterations=iterations,
            invert_p=invert_p,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )
        
#%% Predict -------------------------------------------------------------------

    if predict:
        
        # Path
        path = list(data_path.rglob("*.tif"))[idx]
        
        # Load data
        t0 = time.time()
        print("load : ", end="", flush=False)
        metadata = get_tif_metadata(path)
        C2 = get_tif_data(path, slc="all", chn=1, rscale=True)
        C2 = (C2 // 16).astype("uint8")
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Predict
        unet = UNet(load_name=load_name1)
        t0 = time.time()
        print("predict : ", end="", flush=False)
        prd = (unet.predict(C2, verbose=0) * 255).astype("uint8")
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%%
        
        from skimage.filters import gaussian
        from skimage.segmentation import clear_border
        from skimage.morphology import (
            remove_small_holes, remove_small_objects
            )
        
        # Parameters
        sigma = 0.5
        thresh = 0.5
        remove_border_objects = True

        # Get msk
        prd = gaussian(prd, sigma=sigma, preserve_range=True)
        msk = prd > thresh
        msk = remove_small_holes(msk, area_threshold=1e4)
        msk = remove_small_objects(msk, min_size=1e4)
        if remove_border_objects:
            msk = clear_border(msk)
        
        # Display
        viewer = napari.Viewer()
        viewer.add_image(
            C2, contrast_limits=[0, 255], visible=1,
            gamma=0.5,
            )
        viewer.add_image(
            prd, contrast_limits=[0, 255], visible=0,
            blending="additive", colormap="inferno", opacity=0.5,
            )
        viewer.add_image(
            msk, contrast_limits=[0, 1], visible=1,
            blending="additive", colormap="bop orange", opacity=0.5,
            rendering="attenuated_mip", attenuation=0.5, 
            )
        

#%%
        # def nProcess(
        #         path, 
        #         df=1, 
        #         sigma=2, 
        #         thresh=0.2, 
        #         h=0.15,
        #         min_size=2048,
        #         clear_nBorder=True,
        #         ):
        
        #     # print(f"nProcess() - {path.stem} : ", end="", flush=True)
        #     # t0 = time.time()
        
        #     # Load data
        #     dir_path = data_path / path.stem
        #     prd = io.imread(dir_path / (path.stem + f"_df{df}_predictions.tif"))
            
        #     # Initialize
        #     sigma //= df
        #     min_size //= df  
            
        #     # Segment (watershed)
        #     prd = gaussian(prd, sigma=sigma)
        #     nMask = prd > thresh
        #     nMask = remove_small_objects(nMask, min_size=min_size)
        #     if clear_nBorder:
        #         nMask = clear_border(nMask)
        #     nMarkers = h_maxima(prd, h)
        #     nMarkers[nMask == 0] = 0
        #     nMarkers = label(nMarkers)
        #     nLabels = watershed(-prd, nMarkers, mask=nMask)
                
        #     # Save
        #     io.imsave(
        #         dir_path / (path.stem + f"_df{df}_nLabels.tif"), 
        #         nLabels.astype("uint16"), check_contrast=False
        #         )
            
        #     # t1 = time.time()
        #     # print(f"{t1 - t0:.3f}s")

#%%

        # Display
        viewer = napari.Viewer()
        viewer.add_image(
            C2, contrast_limits=[0, 255], visible=1,
            gamma=0.5,
            )
        viewer.add_image(
            prd, contrast_limits=[0, 255], visible=0,
            blending="additive", colormap="inferno", opacity=0.5,
            )
        viewer.add_image(
            msk, contrast_limits=[0, 1], visible=1,
            blending="additive", colormap="bop orange", opacity=0.5,
            rendering="attenuated_mip", attenuation=0.5, 
            )
