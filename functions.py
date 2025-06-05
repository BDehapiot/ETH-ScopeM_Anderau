#%% Imports -------------------------------------------------------------------

import tifffile
import warnings
import numpy as np

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
    
def get_tif_data(path, slc="all", chn="all"):

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
    
    return arr.squeeze()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    from pathlib import Path
    
    def fetch_info(info, search_str):
        line = [
            line for line in info.splitlines() 
            if line.strip().startswith(search_str)
            ][0]
        return float(line.split(" = ")[1])
    
    # Parameters
    idx, slc, chn = 0, 40, 1
    
    # Path(s)
    data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data")
    paths = list(data_path.rglob("*.tif"))
    path = paths[idx]
    
    # Load metadata & data
    metadata = get_tif_metadata(paths[idx])
    C2 = get_tif_data(path, slc=slc, chn=chn)
    
    pass