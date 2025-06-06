#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

# functions
from functions import get_tif_metadata, get_tif_data

#%% Inputs --------------------------------------------------------------------

# Path(s)
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")

# Parameter(s)
nSlices = 5

#%% Function(s) ---------------------------------------------------------------

def extract():
        
    paths = list(data_path.rglob("*.tif"))
    
    for path in paths:
        
        print(f"extract : {path.stem}")
        
        # Metadata
        metadata = get_tif_metadata(path)
        shape = metadata["shape"]
        
        # Create random z indexes
        z_idxs = np.random.choice(
            np.arange(shape[0]), size=nSlices, replace=False) 
        
        for z_idx in z_idxs:
            img = get_tif_data(path, slc=int(z_idx), chn=1, rscale=True)
            img = norm_pct(img)               
            img = (img * 255).astype("uint8")
            name = path.stem + f"_z{z_idx:03d}.tif"
            io.imsave(
                Path.cwd() / "data" / "train" / name,
                img, check_contrast=False,
                )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    extract()
        
#%% Rescale -------------------------------------------------------------------

    # # skimage
    # from skimage.transform import rescale

    # rfi = 0.065 / 0.2
    # paths = list(Path(Path.cwd(), "data", "train").rglob("*_mask.tif"))
    
    # for path in paths:
    #     msk = io.imread(path)
    #     msk = rescale(msk, (rfi, rfi), order=0)
    #     io.imsave(
    #         str(path).replace("_mask.tif", "_mask_rscale.tif"),
    #         msk, check_contrast=False
    #         )