#%% Imports -------------------------------------------------------------------

import tifffile
import warnings
import numpy as np

# skimage
from skimage.transform import rescale

#%% Function : get_tif_metadata() ---------------------------------------------

def get_tif_metadata(path):
    
    def fetch_info(info, search_str):
        line = [
            line for line in info.splitlines() 
            if line.strip().startswith(search_str)
            ][0]
        return float(line.split(" = ")[1])
    
    with tifffile.TiffFile(str(path)) as tif:
    
        # Shape & axes
        series = tif.series[0]
        shape = series.shape
        axes = series.axes
        
        # IJ metadata
        ij_metadata = tif.imagej_metadata or {}
        info = ij_metadata["Info"]
        voxel_size = (
            fetch_info(info, "dZAxisCalibration"),
            fetch_info(info, "dCalibration"),
            fetch_info(info, "dCalibration"),
            )
        
    return {
        "shape" : shape,
        "axes"  : axes,
        "ij_metadata" : ij_metadata,
        "voxel_size" : voxel_size,
        }
     
#%% Function : get_tif_data() -------------------------------------------------
    
def get_tif_data(path, slc="all", chn="all", rscale=False):
    
    def rescale_data(arr, voxel_size):
        
        # Determine rescaling factors
        rfi = voxel_size[1] / voxel_size[0]
        if arr.ndim == 4:
            rscale = (1, 1, rfi, rfi)
        if arr.ndim == 3:
            rscale = (1, rfi, rfi)
        if arr.ndim == 2:
            rscale = (rfi, rfi) 
            
        # Rescale array
        arr = rescale(arr, rscale, order=0) # iso
        
        return arr

    with tifffile.TiffFile(str(path)) as tif:

        # Metadata
        metadata = get_tif_metadata(path)
        shape = metadata["shape"]
        axes  = metadata["axes"]
        
        # Format inputs
        
        c_dim = axes.find("C")
        if chn == "all":
            channels = np.arange(shape[c_dim])
        elif isinstance(chn, tuple):
            channels = np.array(chn)
        elif isinstance(chn, int):
            channels = np.array([chn])
        else:
            warnings.warn(f"wrong chn request : {type(chn)}", UserWarning)
        
        z_dim = axes.find("Z")
        if slc == "all":
            slices = np.arange(shape[z_dim])
        elif isinstance(slc, tuple):
            slices = np.array(slc)
        elif isinstance(slc, int):
            slices = np.array([slc])
        else:
            warnings.warn(f"wrong slc request : {type(slc)}", UserWarning)
        
        # Load data
        
        arr = np.zeros(
            (len(slices), len(channels), shape[-2], shape[-1]),
            dtype="uint16"
            )
        
        for z in range(len(slices)):
            for c in range(len(channels)):
                arr[z, c, ...] = tif.pages[
                    slices[z] * shape[1] + channels[c]].asarray()
        arr = arr.squeeze()
        
        if rscale:
            arr = rescale_data(arr, metadata["voxel_size"])
    
    return arr
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import napari
    from pathlib import Path
    from bdtools.norm import norm_pct
    
    # Parameters
    idx, slc = 12, "all" 
    
    # Path(s)
    data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")
    paths = list(data_path.rglob("*.tif"))
    path = paths[idx]
    
    # Load metadata & data
    metadata = get_tif_metadata(paths[idx])
    C1 = get_tif_data(path, slc=slc, chn=0)
    C2 = get_tif_data(path, slc=slc, chn=1)
    C3 = get_tif_data(path, slc=slc, chn=2)
    
    # Convert to uint8
    C1 = (norm_pct(C1) * 255).astype("uint8")
    C2 = (norm_pct(C2) * 255).astype("uint8")
    C3 = (norm_pct(C3) * 255).astype("uint8")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(
        C1, visible=0, name="C1",
        # scale=metadata["voxel_size"],
        contrast_limits=[0, 255],
        )
    viewer.add_image(
        C2, visible=0, name="C2",
        # scale=metadata["voxel_size"],
        contrast_limits=[0, 255],
        )
    viewer.add_image(
        C3, visible=0, name="C3",
        # scale=metadata["voxel_size"],
        contrast_limits=[0, 255],
        )
    
    # viewer.add_image(
    #     mrg, visible=1,
    #     scale=metadata["voxel_size"],
    #     contrast_limits=[0, 4095],
    #     )

    pass