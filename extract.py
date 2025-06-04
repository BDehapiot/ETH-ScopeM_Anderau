#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct

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
        
        with tifffile.TiffFile(str(path)) as tif:

            # Get shape & axes
            series = tif.series[0]
            shape = series.shape
            
            # Create random z indexes
            z_idxs = np.random.choice(
                np.arange(shape[0]), size=nSlices, replace=False) 
        
            # Load and save
            for z_idx in z_idxs:
                # C1 = tif.pages[z_idx * shape[1] + 0].asarray()
                C2 = tif.pages[z_idx * shape[1] + 1].asarray()
                # img = (norm_pct(C1) + norm_pct(C2)) / 2
                img = norm_pct(C2)               
                img = (img * 255).astype("uint8")
                name = path.stem + f"_z{z_idx:03d}.tif"
                io.imsave(
                    Path.cwd() / "data" / "train" / name,
                    img, check_contrast=False,
                    )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    extract()