#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# functions
from functions import check_nd2, import_nd2

#%% Inputs --------------------------------------------------------------------

# Path(s)
data_path = Path("D:\local_Anderau\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")

# Parameter(s)
nSlices = 5

#%% Function(s) ---------------------------------------------------------------

def extract(path):
    
    print(f"extract : {path.stem}")
    
    # Metadata
    shape = check_nd2(path)
    
    # Create random z indexes
    z_idxs = np.random.choice(
        np.arange(shape[0]), size=nSlices, replace=False) 
    
    for z_idx in z_idxs:
        arr, metadata = import_nd2(path, z=int(z_idx), c=1, rscale=True)
        name = path.stem + f"_z{z_idx:03d}.tif"
        io.imsave(
            Path.cwd() / "data" / "train" / name,
            arr, check_contrast=False,
            )
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    paths = list(data_path.glob("*.nd2"))
    for path in paths:
        extract(path)