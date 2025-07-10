#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import import_nd2

# bdtools
from bdtools.models.annotate import Annotate
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Path
data_path = Path("D:\local_Anderau\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")
train_path = Path("data", "train")

# Procedure
annotate = 1
train = 0
predict = 0

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
load_name1 = "model_256_edt_4000-468_1"

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
        path = list(data_path.glob("*.nd2"))[idx]
        
        # Load data
        t0 = time.time()
        print("import_nd2() : ", end="", flush=False)
        C2, _ = import_nd2(path, z="all", c=1, rscale=True)
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Predict
        unet = UNet(load_name=load_name1)
        t0 = time.time()
        print("predict : ", end="", flush=False)
        prd = (unet.predict(C2, verbose=0) * 255).astype("uint8")
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
            
        # Display
        vwr = napari.Viewer()
        vwr.add_image(C2)
        vwr.add_image(prd)        

