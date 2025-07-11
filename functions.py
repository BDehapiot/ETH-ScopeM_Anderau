#%% Imports -------------------------------------------------------------------

import nd2
import warnings
import numpy as np

# skimage
from skimage.transform import rescale

#%% Function : check_nd2() ----------------------------------------------------

def check_nd2(path):
    
    # Initialize
    warnings.filterwarnings(
        "ignore", 
        message="ND2File file not closed before garbage collection"
        )
    
    with nd2.ND2File(path) as f:
        darr = f.to_dask()
    
    return darr.shape

#%% Function : import_nd2() ---------------------------------------------------

def import_nd2(path, z="all", c="all", rscale=True):
    
    # Initialize
    zi = slice(None) if z == "all" else z 
    ci = slice(None) if c == "all" else c 
        
    with nd2.ND2File(path) as f:
        
        # Input voxel size
        vsize0 = (
            f.voxel_size()[2],
            f.voxel_size()[1],
            f.voxel_size()[0],
            )
        
        # Load        
        darr = f.to_dask()
        arr  = darr[zi, ci, ...].compute()

        
    # Iso. rescaling factor (rfi)
    rfi = vsize0[1] / vsize0[0]
    if arr.ndim == 4:
        rscale = (1, 1, rfi, rfi)
    if arr.ndim == 3 and z == "all":
        rscale = (1, rfi, rfi)
    if arr.ndim == 3 and c == "all":
        rscale = (1, rfi, rfi)
    if arr.ndim == 2:
        rscale = (rfi, rfi)
        
    # Rescale array
    shape0 = arr.shape
    arr = rescale(arr, rscale, order=0) # iso
    shape1 = arr.shape 
    
    # Convert to "uint8" (from 0-4095 to 0-255)
    arr = (arr // 16)
    arr = np.clip(arr, 0, 255)
    arr = arr.astype("uint8")
    
    # Metadata    
    cond = path.stem.split("_")[0]
    time = int(path.stem.split("_")[1][:2])
    repl = int(path.stem.split("_")[2])
    metadata = {
        "path"   : path,
        "cond"   : cond,
        "time"   : time,
        "repl"   : repl,
        "shape0" : shape0,
        "shape1" : shape1,
        "vsize0" : vsize0,
        "vsize1" : (vsize0[0],) * 3,
        }

    return arr, metadata

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":   
    
    pass
    
#%% Test : import_nd2() -------------------------------------------------------

    # Imports
    import time
    import napari
    from pathlib import Path
    
    # Parameters
    z, c = "all", "all" 
    
    # Path(s)
    data_path = Path("D:\local_Anderau\data")
    # data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")
    paths = list(data_path.glob("*.nd2"))
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("import_nd2() :", end="", flush=False)
    
    arr, metadata = import_nd2(paths[0], z="all", c="all")
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(arr)
    
#%% Import & save all ---------------------------------------------------------

    # # Imports
    # import time
    # from skimage import io
    # from pathlib import Path
    
    # # Parameters
    # z, c = "all", "all" 
    
    # # Path(s)
    # data_path = Path("D:\local_Anderau\data")
    # paths = list(data_path.glob("*.nd2"))
    
    # # -------------------------------------------------------------------------
    
    # # for path in paths:
                
    # path = paths[-1]
        
    # t0 = time.time()
    # print(f"import & save - {path.name} : ", end="", flush=False)
    
    # arr, metadata = import_nd2(path, z="all", c="all")
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # Save
    # for i in range(arr.shape[1]):
    #     save_path = data_path / (path.stem + f"_C{i + 1}.tif")
    #     io.imsave(save_path, arr[:, i, ...], check_contrast=False)
