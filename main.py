#%% Imports -------------------------------------------------------------------

import time
import pickle
import shutil
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
# from joblib import Parallel, delayed

# bdtools
from bdtools.models.unet import UNet
from bdtools import nan_filt, nan_replace

# functions
from functions import import_nd2

# Skimage
from skimage.measure import label
from skimage.filters import gaussian
from skimage.filters.rank import median
from skimage.transform import resize, rescale
from skimage.morphology import (
    disk, ball, binary_erosion, remove_small_holes, remove_small_objects
    )
from skimage.segmentation import (
    clear_border, watershed, find_boundaries, expand_labels, relabel_sequential
    )

# Scipy
from scipy.ndimage import mean

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

# Napari
import napari

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#%% Inputs --------------------------------------------------------------------

# Procedure
procedure = {
    
    "extract" : 0,
    "predict" : 0,
    "process" : 0,
    "analyse" : 1,
    "display" : 0,
    
    }

# Parameters
parameters = {
    
    # Paths
    "data_path" :
        Path("D:\local_Anderau\data"),
    # "data_path" :
    #     Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data"),
    "model_name" :
        "model_256_edt_4000-648_1",
        
    # Process
    
        # Get labels
        "sigma0"    : 0.5,
        "sigma1"    : 2,
        "thresh0"   : 0.01,
        "thresh1"   : 0.5,
        "remove_border_objects" : False,
        "min_sdist" : 5, # in pixels (of rescaled data)
    
        # Get intensities
        "mbn_width" : 3, # in pixels (of rescaled data)
        "bsub"      : True,

    # Display
    "C1_contrast_limits" : [0, 100],
    "C2_contrast_limits" : [0, 100],
    "C3_contrast_limits" : [0, 100],
    
    }

#%% Class(Main) ---------------------------------------------------------------

class Main:
    
    def __init__(
        self, 
        procedure=procedure, 
        parameters=parameters,
        ):
    
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.data_path  = parameters["data_path"] 
        self.model_name = parameters["model_name"]
        
        # Initialize
        self.paths = list(self.data_path.glob("*.nd2"))
                
        # Run
        if self.procedure["extract"]:
            self.extract() 
        if self.procedure["predict"]:
            self.predict() 
        if self.procedure["process"]:
            self.process() 
        if self.procedure["analyse"]:
            self.analyse() 

#%% Class(Main) : extract() ---------------------------------------------------

    def extract(self):
        
        def _extract(path):
            
            # Setup directory
            out_path = path.parent / path.stem 
            C1_path = out_path / "C1.tif"
            if out_path.exists():
                if self.procedure["extract"] == 1 and C1_path.exists():
                    return
                elif self.procedure["extract"] == 2:
                    for item in out_path.iterdir():
                        if item.is_file() or item.is_symlink():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
            else:
                out_path.mkdir(parents=True, exist_ok=True)
            
            # Load
            t0 = time.time()
            print(f"{path.stem}")
            print("load : ", end="", flush=False)
            arr, metadata = import_nd2(path, z="all", c="all", rscale=True)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
            # Save
            t0 = time.time()
            print("save : ", end="", flush=False)
            io.imsave(out_path / "C1.tif", arr[:, 0, ...], check_contrast=False)
            io.imsave(out_path / "C2.tif", arr[:, 1, ...], check_contrast=False)
            io.imsave(out_path / "C3.tif", arr[:, 2, ...], check_contrast=False)
            with open(str(out_path / "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
        # Execute
        for path in self.paths:
            _extract(path)

#%% Class(Main) : predict() ---------------------------------------------------

    def predict(self):
        
        def _predict(path):
            
            out_path = path.parent / path.stem
            prd_path = out_path / "prd.tif"
            
            if prd_path.exists() and self.procedure["predict"] == 1:
            
                return
            
            else:

                # Load
                t0 = time.time()
                print(f"{path.stem}")
                print("load : ", end="", flush=False)
                C2 = io.imread(out_path / "C2.tif")
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
                # Predict
                t0 = time.time()
                print("predict : ", end="", flush=False)
                prd = (unet.predict(C2, verbose=0) * 255).astype("uint8")
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
                # Save
                t0 = time.time()
                print("save : ", end="", flush=False)
                io.imsave(out_path / "prd.tif", prd, check_contrast=False)
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
        
        # Execute
        unet = UNet(load_name=self.model_name)
        for path in self.paths:
            _predict(path)  

#%% Class(Main) : process() ---------------------------------------------------

    def process(self):
        
        def remove_small_obj(lbl, min_size=1e4): # parameter
            vals, counts = np.unique(lbl.ravel(), return_counts=True)
            msk = (vals != 0) & (counts >= min_size)
            valid_vals = vals[msk]
            msk_img = np.isin(lbl, valid_vals)
            lbl_cleaned = np.where(msk_img, lbl, 0)
            return relabel_sequential(lbl_cleaned)[0]
        
        def get_synaptic_plane_dist(C3, med_size=21):
            msk = C3 > 128 # parameter
            msk = remove_small_objects(msk, min_size=1e5) # parameter
            z_pos = msk.shape[0] - np.argmax(msk[::-1, :, :], axis=0)
            z_pos = median(z_pos.astype("uint8"), footprint=disk(med_size))
            sdist = np.arange(msk.shape[0])[:, np.newaxis, np.newaxis]
            sdist = np.broadcast_to(sdist, msk.shape)
            sdist = sdist - z_pos
            return sdist
        
        def subtract_background(Cx, all_lbl, kernel_size=9):       
            msk = all_lbl == 0
            bgrd = Cx.copy().astype(float)
            bgrd *= msk
            bgrd[bgrd == 0] = np.nan
            bgrd = rescale(bgrd, 0.25, order=0)
            bgrd = nan_filt(bgrd, kernel_size=kernel_size)
            bgrd = nan_replace(bgrd, kernel_size=kernel_size)
            bgrd = resize(bgrd, Cx.shape, order=0)
            bsub = Cx.copy().astype(float) - bgrd
            return bsub.astype("float32")
        
        def get_mean_int_profiles(
                Cx, sdist, x_lbl, bin_size=1, min_bin=-10, max_bin=100):
            sdist_prf = sdist[x_lbl > 0]
            Cx_prf = Cx[x_lbl > 0]
            if min_bin is None:
                min_bin = np.min(sdist_prf)
            if max_bin is None:
                max_bin = np.max(sdist_prf)
            bins = np.arange(min_bin, max_bin + bin_size, bin_size)
            bin_idxs = np.digitize(sdist_prf, bins) - 1
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mean_int_prf = np.full(len(bin_centers), np.nan)
            for i in range(len(bin_centers)):
                in_bin = bin_idxs == i
                if np.any(in_bin):
                    mean_int_prf[i] = np.mean(Cx_prf[in_bin])
            return np.column_stack((bin_centers, mean_int_prf))
        
        def get_obj_volume(lbl):
            vals, counts = np.unique(lbl.ravel(), return_counts=True)
            return counts[1:]
                        
        def get_obj_mean_int(lbl, img):
            labels = np.unique(lbl)[1:]
            return mean(img, labels=lbl, index=labels) 

        def _process(path):
            
            out_path = path.parent / path.stem
            lbl_path = out_path / "lbl.tif"

            if lbl_path.exists() and self.procedure["process"] == 1:
            
                return
            
            else:

                # Load
                print(f"{path.stem}")
                
                t0 = time.time()
                print("load    : ", end="", flush=False)
                C1 = io.imread(out_path / "C1.tif")
                C2 = io.imread(out_path / "C2.tif")
                C3 = io.imread(out_path / "C3.tif")
                prd = io.imread(out_path / "prd.tif")
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
                t0 = time.time()
                print("Process : ", end="", flush=False)
            
                # Get labels
                msk0 = gaussian(prd, sigma=sigma0, preserve_range=True) > thresh0
                msk0 = remove_small_holes(msk0, area_threshold=1e4) # parameter
                if remove_border_objects:
                    msk0 = clear_border(msk0)
                msk1 = gaussian(prd, sigma=sigma1, preserve_range=True) > thresh1
                mrk = label(msk1)
                lbl = watershed(-prd, mrk, mask=msk0).astype("uint8")
                sdist = get_synaptic_plane_dist(C3, med_size=21) # parameter
                lbl[find_boundaries(lbl)] = 0
                lbl = remove_small_obj(lbl, min_size=1e4)
                                
                # Get cyt, mbn and all labels
                cyt_msk = binary_erosion(lbl > 0, footprint=ball(max(3, mbn_width)))
                cyt_lbl = lbl.copy()
                cyt_lbl[~cyt_msk] = 0
                cyt_lbl = cyt_lbl.astype("uint8")
                mbn_lbl = expand_labels(lbl, distance=max(3, mbn_width))
                mbn_lbl[cyt_msk] = 0
                mbn_lbl = mbn_lbl.astype("uint8")
                all_lbl = np.maximum(cyt_lbl, mbn_lbl)
                    
                # Background subtraction (optional)
                if bsub:
                    C1 = subtract_background(C1, all_lbl, kernel_size=9)
                    C2 = subtract_background(C2, all_lbl, kernel_size=9)       
                
                # Mean intensity profile (mean int vs. sdist)                
                C1_cyt_mean_prf = get_mean_int_profiles(C1, sdist, cyt_lbl)
                C2_cyt_mean_prf = get_mean_int_profiles(C2, sdist, cyt_lbl)
                C1_mbn_mean_prf = get_mean_int_profiles(C1, sdist, mbn_lbl)
                C2_mbn_mean_prf = get_mean_int_profiles(C2, sdist, mbn_lbl)
                C1_all_mean_prf = get_mean_int_profiles(C1, sdist, all_lbl)
                C2_all_mean_prf = get_mean_int_profiles(C2, sdist, all_lbl)
                
                # Trim lbl masks (acc. to sdist)
                cyt_lbl[sdist < min_sdist] = 0
                mbn_lbl[sdist < min_sdist] = 0
                all_lbl[sdist < min_sdist] = 0
                        
                # Get results
                volume_cyt  = get_obj_volume(cyt_lbl)
                volume_mbn  = get_obj_volume(mbn_lbl)
                volume_all  = get_obj_volume(all_lbl)
                C1_cyt_mean = get_obj_mean_int(cyt_lbl, C1)
                C2_cyt_mean = get_obj_mean_int(cyt_lbl, C2)
                C1_mbn_mean = get_obj_mean_int(mbn_lbl, C1)
                C2_mbn_mean = get_obj_mean_int(mbn_lbl, C2)
                C1_all_mean = get_obj_mean_int(all_lbl, C1)
                C2_all_mean = get_obj_mean_int(all_lbl, C2)
                results = {
                    "name"        : path.stem,
                    "label"       : np.arange(1, np.max(lbl) + 1),
                    "volume_cyt"  : volume_cyt,
                    "volume_mbn"  : volume_mbn,
                    "volume_all"  : volume_all,
                    "C1_cyt_mean" : C1_cyt_mean, "C2_cyt_mean" : C2_cyt_mean,
                    "C1_mbn_mean" : C1_mbn_mean, "C2_mbn_mean" : C2_mbn_mean,
                    "C1_all_mean" : C1_all_mean, "C2_all_mean" : C2_all_mean,
                    "C1_cyt_sum"  : C1_cyt_mean * volume_cyt,
                    "C2_cyt_sum"  : C2_cyt_mean * volume_cyt,
                    "C1_mbn_sum"  : C1_mbn_mean * volume_mbn,
                    "C2_mbn_sum"  : C2_mbn_mean * volume_mbn,
                    "C1_all_sum"  : C1_all_mean * volume_all,
                    "C2_all_sum"  : C2_all_mean * volume_all,
                    "bin_centers_prf" : C1_cyt_mean_prf[:, 0],
                    "C1_cyt_mean_prf" : C1_cyt_mean_prf[:, 1],
                    "C2_cyt_mean_prf" : C2_cyt_mean_prf[:, 1],
                    "C1_mbn_mean_prf" : C1_mbn_mean_prf[:, 1],
                    "C2_mbn_mean_prf" : C2_mbn_mean_prf[:, 1],
                    "C1_all_mean_prf" : C1_all_mean_prf[:, 1],
                    "C2_all_mean_prf" : C2_all_mean_prf[:, 1],
                    }
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                    
                # Save
                t0 = time.time()
                print("save    : ", end="", flush=False)
                
                # pkl
                with open(out_path / "results.pkl", "wb") as f:
                    pickle.dump(results, f)
                
                # csv
                results_df = pd.DataFrame(
                    {k: v for k, v in results.items() if "prf" not in k})   
                results_prf_df = pd.DataFrame(
                    {k: v for k, v in results.items() if "prf" in k})   
                results_df.to_csv(out_path / "results.csv", index=False)
                results_prf_df.to_csv(out_path / "results_prf.csv", index=False)
                
                # tif
                io.imsave(out_path / "lbl.tif", lbl, check_contrast=False)
                io.imsave(out_path / "cyt_lbl.tif", cyt_lbl, check_contrast=False)
                io.imsave(out_path / "mbn_lbl.tif", mbn_lbl, check_contrast=False)
                io.imsave(out_path / "all_lbl.tif", all_lbl, check_contrast=False)
                if bsub:
                    io.imsave(out_path / "C1_bsub.tif", C1, check_contrast=False)
                    io.imsave(out_path / "C2_bsub.tif", C2, check_contrast=False)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
        # Fetch 
        sigma0   = self.parameters["sigma0"]
        sigma1   = self.parameters["sigma1"]
        thresh0  = self.parameters["thresh0"] * 255
        thresh1  = self.parameters["thresh1"] * 255
        remove_border_objects = self.parameters["remove_border_objects"]
        min_sdist = self.parameters["min_sdist"]
        mbn_width = self.parameters["mbn_width"]
        bsub = self.parameters["bsub"]
        
        # Execute
        
        for path in self.paths:
            _process(path)
            
        # Parallel(n_jobs=-1)(
        #     delayed(_process)(path)
        #     for path in self.paths
        #     )
        
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        def filter_data(df, tags):
            if tags:
                mask = df["name"].apply(lambda x: all(tag in x for tag in tags))
            else:
                mask = pd.Series(True, index=df.index)
            return df.loc[mask]
        
        def prepare_data():
            
            # Load, merge, group & save results
            self.results_m = []
            for path in self.paths:
                out_path = path.parent / path.stem
                self.results_m.append(pd.read_csv(out_path / "results.csv"))
            self.results_m = pd.concat(self.results_m, ignore_index=True)
            self.results_m_g = self.results_m.groupby("name").mean(numeric_only=True).reset_index()
            for i in self.results_m_g.index:        
                self.results_m_g.loc[i, "label"] = len(
                    filter_data(self.results_m, [self.results_m_g.loc[i, "name"]]))
            self.results_m_g = self.results_m_g.rename(columns={"label": "nlabel"})
            self.results_m.to_csv(self.data_path / "results_m.csv", index=False)
            self.results_m_g.to_csv(self.data_path / "results_m_g.csv", index=False) 
            
            # Load & merge results_prf
            self.results_prf_m = []
            for path in self.paths:
                out_path = path.parent / path.stem
                self.results_prf_m.append((
                    path.stem, pd.read_csv(out_path / "results_prf.csv")))
                
        def plot_data(channel, cond, timepoints):
            
            fig = plt.figure(figsize=(9, 9), layout="tight")
            gs = GridSpec(3, 3, figure=fig)
                    
            # Metrics & axis mapping 
            metrics = [
                ("cyt_mean"    , (0, 0)),
                ("mbn_mean"    , (1, 0)),
                ("all_mean"    , (2, 0)),
                ("cyt_sum"     , (0, 1)),
                ("mbn_sum"     , (1, 1)),
                ("all_sum"     , (2, 1)),
                ("cyt_mean_prf", (0, 2)),
                ("mbn_mean_prf", (1, 2)),
                ("all_mean_prf", (2, 2)),
                ]
            
            # Axes dict.
            axes = {
                metric: fig.add_subplot(gs[row, col]) 
                for metric, (row, col) in metrics
                }
                
            # Plot dots + bars
            for m, (metric, ax) in enumerate(axes.items()):
                key = f"{channel}_{metric}"
                if m < 6:
                    for t, tp in enumerate(timepoints):
                        data = filter_data(self.results_m_g, [cond, tp])[key]
                        ax.bar((t,) * len(data), np.mean(data), color="lightgray")
                        ax.scatter((t,) * len(data), data)
                    ax.set_title(f"({channel}) {cond}_" + metric)
                    ax.set_xticks([0, 1, 2])
                    ax.set_xticklabels(timepoints)     
            
            # Plot profiles
            for m, (metric, ax) in enumerate(axes.items()):
                key = f"{channel}_{metric}"
                bin_centers = np.array(self.results_prf_m[0][1]["bin_centers_prf"])
                if m >= 6:
                    for t, tp in enumerate(timepoints):
                        data = []
                        for r in self.results_prf_m: 
                            if f"{cond}_{tp}" in r[0]:
                                data.append(np.array(r[1][key]))
                        ax.plot(bin_centers, np.mean(np.stack(data), axis=0))
                    ax.axvline(0, color="k", linestyle="--", linewidth=1)
                    ax.set_title(f"({channel}) {cond}_" + metric)
                    ax.set_xticks([0, 25, 50, 75])
                    ax.set_xlabel("dist. to synaptic plane (pix)")   
                    
             # Save
            plt.savefig(self.data_path / (f"plot_{channel}_{cond}.png"), format="png")
                    
        prepare_data()
        timepoints = ["00min", "15min", "30min"] 
        plot_data("C1", "PEG12", timepoints)
        plot_data("C2", "PEG12", timepoints)
        plot_data("C1", "PEG34", timepoints)
        plot_data("C2", "PEG34", timepoints)
    
#%% Class(Display) ------------------------------------------------------------

class Display:
    
    def __init__(
            self, 
            procedure=procedure, 
            parameters=parameters,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.data_path  = parameters["data_path"] 
        
        # Initialize
        self.idx = 0
        self.paths = list(self.data_path.glob("*.nd2"))
        
        # Run
        if self.procedure["display"]:
            self.init_viewer()
            
#%% Class(Display) : function(s) ----------------------------------------------

    def load_data(self):  
        
        path = self.paths[self.idx]
        out_path = path.parent / path.stem
        
        self.C1  = io.imread(out_path / "C1.tif")
        self.C2  = io.imread(out_path / "C2.tif")
        self.C3  = io.imread(out_path / "C3.tif")
        self.prd = io.imread(out_path / "prd.tif")
        self.lbl = io.imread(out_path / "lbl.tif")
        self.mbn_lbl = io.imread(out_path / "mbn_lbl.tif")
        
        self.mbn_out = self.mbn_lbl > 0
        self.mbn_out = self.mbn_out ^ binary_erosion(self.mbn_out)

    def next_hstack(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self.update()
            
    def prev_hstack(self):
        if self.idx > 0:
            self.idx -= 1
            self.update()
            
    def show_htk(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C1", "C2", "C3"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.dims.ndisplay = 2
        self.viewer.grid.enabled = True
    
    def show_prd(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C2", "prd"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.dims.ndisplay = 3
        self.viewer.grid.enabled = False
        
    def show_lbl(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["C2", "lbl"]:
                self.viewer.layers[name].visible = 1
            else:
                self.viewer.layers[name].visible = 0
        self.viewer.dims.ndisplay = 3
        self.viewer.grid.enabled = False
        
    def show_chk(self):
        for name in self.viewer.layers:
            name = str(name)
            if self.rad_C1.isChecked():
                if name in ["C1", "mbn_out"]:
                    self.viewer.layers[name].visible = 1
                else:
                    self.viewer.layers[name].visible = 0
            if self.rad_C2.isChecked():
                if name in ["C2", "mbn_out"]:
                    self.viewer.layers[name].visible = 1
                else:
                    self.viewer.layers[name].visible = 0
            if self.rad_C3.isChecked():
                if name in ["C3", "mbn_out"]:
                    self.viewer.layers[name].visible = 1
                else:
                    self.viewer.layers[name].visible = 0
        self.viewer.dims.ndisplay = 2
        self.viewer.grid.enabled = False
        
    def hide_layers(self):
        if self.rad_prd.isChecked():
            self.viewer.layers["prd"].visible = 0
        if self.rad_lbl.isChecked():
            self.viewer.layers["lbl"].visible = 0
        if (self.rad_C1.isChecked() or 
            self.rad_C2.isChecked() or 
            self.rad_C3.isChecked() 
            ):
            self.viewer.layers["mbn_out"].visible = 0

    def show_layers(self):
        if self.rad_prd.isChecked():
            self.viewer.layers["prd"].visible = 1
        if self.rad_lbl.isChecked():
            self.viewer.layers["lbl"].visible = 1
        if (self.rad_C1.isChecked() or 
            self.rad_C2.isChecked() or 
            self.rad_C3.isChecked() 
            ):
            self.viewer.layers["mbn_out"].visible = 1
            
#%% Class(Display) : init_viewer() --------------------------------------------                

    def init_viewer(self):
                
        # Create viewer
        self.viewer = napari.Viewer()
        
        # Create "hstack" menu
        self.htk_group_box = QGroupBox("Select hstack")
        htk_group_layout = QVBoxLayout()
        self.btn_next_htk = QPushButton("next")
        self.btn_prev_htk = QPushButton("prev")
        htk_group_layout.addWidget(self.btn_next_htk)
        htk_group_layout.addWidget(self.btn_prev_htk)
        self.htk_group_box.setLayout(htk_group_layout)
        self.btn_next_htk.clicked.connect(self.next_hstack)
        self.btn_prev_htk.clicked.connect(self.prev_hstack)
        
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QVBoxLayout()
        row1_layout = QHBoxLayout()
        self.rad_htk = QRadioButton("hstack")
        self.rad_prd = QRadioButton("predictions")
        self.rad_lbl = QRadioButton("labels")
        self.rad_htk.setChecked(True)
        row1_layout.addWidget(self.rad_htk)
        row1_layout.addWidget(self.rad_prd)
        row1_layout.addWidget(self.rad_lbl)
        row2_layout = QHBoxLayout()
        self.rad_C1 = QRadioButton("C1")
        self.rad_C2 = QRadioButton("C2")
        self.rad_C3 = QRadioButton("C3")
        row2_layout.addWidget(self.rad_C1)
        row2_layout.addWidget(self.rad_C2)
        row2_layout.addWidget(self.rad_C3)
        dsp_group_layout.addLayout(row1_layout)
        dsp_group_layout.addLayout(row2_layout)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_htk.toggled.connect(
            lambda checked: self.show_htk() if checked else None)
        self.rad_prd.toggled.connect(
            lambda checked: self.show_prd() if checked else None)
        self.rad_lbl.toggled.connect(
            lambda checked: self.show_lbl() if checked else None)   
        self.rad_C1.toggled.connect(
            lambda checked: self.show_chk() if checked else None)
        self.rad_C2.toggled.connect(
            lambda checked: self.show_chk() if checked else None)
        self.rad_C3.toggled.connect(
            lambda checked: self.show_chk() if checked else None)
        
        # Create texts
        self.info = QLabel()
        self.info.setFont(QFont("Consolas"))
        self.info.setText(self.get_info())

        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.htk_group_box)
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        self.init_layers()    
        self.show_htk()
        
        # Shortcuts
        
        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_hstack()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_hstack()
        
        @self.viewer.bind_key("Enter", overwrite=True)
        def toogle_layers(viewer):
            self.hide_layers()
            yield
            self.show_layers()
            
#%% Class(Display) : init_layers() --------------------------------------------

    def init_layers(self):  
        
        self.load_data()
        
        # out
        self.viewer.add_image(
            self.mbn_out, name="mbn_out", visible=0,
            colormap="gray", blending="additive", 
            gamma=1.0, opacity=0.25,
            )
        
        # lbl
        self.viewer.add_labels(
            self.lbl, name="lbl", visible=0,
            blending="additive", 
            opacity=0.50,
            )
        
        # prd
        self.viewer.add_image(
            self.prd, name="prd", visible=0,
            colormap="inferno", blending="additive", 
            gamma=1.0, opacity=0.25,
            )
        
        # htk
        self.viewer.add_image(
            self.C3, name="C3", visible=1,
            colormap="gray", blending="additive", 
            gamma=1.0, opacity=1.00,
            contrast_limits = self.parameters["C3_contrast_limits"],
            )
        self.viewer.add_image(
            self.C2, name="C2", visible=1,
            colormap="gray", blending="additive", 
            gamma=1.0, opacity=1.00,
            contrast_limits = self.parameters["C2_contrast_limits"],
            )
        self.viewer.add_image(
            self.C1, name="C1", visible=1,
            colormap="gray", blending="additive",  
            gamma=1.0, opacity=1.00,
            contrast_limits = self.parameters["C1_contrast_limits"],
            )

#%% Class(Display) : get_info() -----------------------------------------------

    def get_info(self):
        
        path = self.paths[self.idx]
        
        return (
    
            f"{path.stem}\n"
            "\n"
            f"prev/next     : page up/down\n"
            f"hide layer(s) : enter\n"
        
            )

#%% Class(Display) : update() -------------------------------------------------

    def update(self):
        
        self.load_data()
        
        # out
        self.viewer.layers["mbn_out"].data = self.mbn_out 
        
        # lbl
        self.viewer.layers["lbl"].data = self.lbl 
       
        # prd
        self.viewer.layers["prd"].data = self.prd 
       
        # htk
        self.viewer.layers["C1"].data = self.C1
        self.viewer.layers["C2"].data = self.C2
        self.viewer.layers["C3"].data = self.C3

        # info
        self.info.setText(self.get_info())
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main()
    display = Display()
    
#%% 
    
    # idx = 5
    # data_path = parameters["data_path"]
    # paths = list(data_path.glob("*.nd2"))
    # path = paths[idx]
    # out_path = path.parent / path.stem
    # C1 = io.imread(out_path / "C1.tif")
    # C2 = io.imread(out_path / "C2.tif")
    # C3 = io.imread(out_path / "C3.tif")
    # prd = io.imread(out_path / "prd.tif")
    
    # # -------------------------------------------------------------------------
    
    # # Parameters
    # sigma0  = parameters["sigma0"]
    # sigma1  = parameters["sigma1"]
    # thresh0 = parameters["thresh0"] * 255
    # thresh1 = parameters["thresh1"] * 255
    # remove_border_objects = parameters["remove_border_objects"]
    # min_sdist = parameters["min_sdist"]
    # mbn_width = parameters["mbn_width"]
    # bsub = parameters["bsub"]

    # # -------------------------------------------------------------------------

    # def remove_small_obj(lbl, min_size=1e4): # parameter
    #     vals, counts = np.unique(lbl.ravel(), return_counts=True)
    #     msk = (vals != 0) & (counts >= min_size)
    #     valid_vals = vals[msk]
    #     msk_img = np.isin(lbl, valid_vals)
    #     lbl_cleaned = np.where(msk_img, lbl, 0)
    #     return relabel_sequential(lbl_cleaned)[0]
    
    # def get_synaptic_plane_dist(C3, med_size=21):
    #     msk = C3 > 128 # parameter
    #     msk = remove_small_objects(msk, min_size=1e5) # parameter
    #     z_pos = msk.shape[0] - np.argmax(msk[::-1, :, :], axis=0)
    #     z_pos = median(z_pos.astype("uint8"), footprint=disk(med_size))
    #     sdist = np.arange(msk.shape[0])[:, np.newaxis, np.newaxis]
    #     sdist = np.broadcast_to(sdist, msk.shape)
    #     sdist = sdist - z_pos
    #     return sdist
    
    # def subtract_background(Cx, all_lbl, kernel_size=9):       
    #     msk = all_lbl == 0
    #     bgrd = Cx.copy().astype(float)
    #     bgrd *= msk
    #     bgrd[bgrd == 0] = np.nan
    #     bgrd = rescale(bgrd, 0.25, order=0)
    #     bgrd = nan_filt(bgrd, kernel_size=kernel_size)
    #     bgrd = nan_replace(bgrd, kernel_size=kernel_size)
    #     bgrd = resize(bgrd, Cx.shape, order=0)
    #     bsub = Cx.copy().astype(float) - bgrd
    #     return bsub.astype("float32")
    
    # def get_mean_int_profiles(
    #         Cx, sdist, x_lbl, bin_size=1, min_bin=-10, max_bin=100):
    #     sdist_prf = sdist[x_lbl > 0]
    #     Cx_prf = Cx[x_lbl > 0]
    #     if min_bin is None:
    #         min_bin = np.min(sdist_prf)
    #     if max_bin is None:
    #         max_bin = np.max(sdist_prf)
    #     bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    #     bin_idxs = np.digitize(sdist_prf, bins) - 1
    #     bin_centers = (bins[:-1] + bins[1:]) / 2
    #     mean_int_prf = np.full(len(bin_centers), np.nan)
    #     for i in range(len(bin_centers)):
    #         in_bin = bin_idxs == i
    #         if np.any(in_bin):
    #             mean_int_prf[i] = np.mean(Cx_prf[in_bin])
    #     return np.column_stack((bin_centers, mean_int_prf))
    
    # def get_obj_volume(lbl):
    #     vals, counts = np.unique(lbl.ravel(), return_counts=True)
    #     return counts[1:]
                    
    # def get_obj_mean_int(lbl, img):
    #     labels = np.unique(lbl)[1:]
    #     return mean(img, labels=lbl, index=labels) 

    # # -------------------------------------------------------------------------
    
    # t0 = time.time()
    # print("get labels : ", end="", flush=False)

    # # Get labels
    # msk0 = gaussian(prd, sigma=sigma0, preserve_range=True) > thresh0
    # msk0 = remove_small_holes(msk0, area_threshold=1e4) # parameter
    # if remove_border_objects:
    #     msk0 = clear_border(msk0)
    # msk1 = gaussian(prd, sigma=sigma1, preserve_range=True) > thresh1
    # mrk = label(msk1)
    # lbl = watershed(-prd, mrk, mask=msk0).astype("uint8")
    # sdist = get_synaptic_plane_dist(C3, med_size=21) # parameter
    # lbl[find_boundaries(lbl)] = 0
    # lbl = remove_small_obj(lbl, min_size=1e4)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
     
    # t0 = time.time()
    # print("process labels : ", end="", flush=False)
    
    # # Get cyt, mbn and all labels
    # cyt_msk = binary_erosion(lbl > 0, footprint=ball(max(3, mbn_width)))
    # cyt_lbl = lbl.copy()
    # cyt_lbl[~cyt_msk] = 0
    # cyt_lbl = cyt_lbl.astype("uint8")
    # mbn_lbl = expand_labels(lbl, distance=max(3, mbn_width))
    # mbn_lbl[cyt_msk] = 0
    # mbn_lbl = mbn_lbl.astype("uint8")
    # all_lbl = np.maximum(lbl, mbn_lbl)
        
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
        
    # t0 = time.time()
    # print("background sub. : ", end="", flush=False)
    
    # if bsub:
    #     C1 = subtract_background(C1, all_lbl, kernel_size=9)
    #     C2 = subtract_background(C2, all_lbl, kernel_size=9)       
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # t0 = time.time()
    # print("mean_int_profiles : ", end="", flush=False)
    
    # C1_cyt_mean_prf = get_mean_int_profiles(C1, sdist, cyt_lbl)
    # C2_cyt_mean_prf = get_mean_int_profiles(C2, sdist, cyt_lbl)
    # C1_mbn_mean_prf = get_mean_int_profiles(C1, sdist, mbn_lbl)
    # C2_mbn_mean_prf = get_mean_int_profiles(C2, sdist, mbn_lbl)
    # C1_all_mean_prf = get_mean_int_profiles(C1, sdist, all_lbl)
    # C2_all_mean_prf = get_mean_int_profiles(C2, sdist, all_lbl)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # t0 = time.time()
    # print("trim. sdist : ", end="", flush=False)

    # cyt_lbl[sdist < min_sdist] = 0
    # mbn_lbl[sdist < min_sdist] = 0
    # all_lbl[sdist < min_sdist] = 0

    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")

    # t0 = time.time()
    # print("measure : ", end="", flush=False)

    # # Get results
    # volume_cyt  = get_obj_volume(cyt_lbl)
    # volume_mbn  = get_obj_volume(mbn_lbl)
    # volume_all  = get_obj_volume(all_lbl)
    # C1_cyt_mean = get_obj_mean_int(cyt_lbl, C1)
    # C2_cyt_mean = get_obj_mean_int(cyt_lbl, C2)
    # C1_mbn_mean = get_obj_mean_int(mbn_lbl, C1)
    # C2_mbn_mean = get_obj_mean_int(mbn_lbl, C2)
    # C1_all_mean = get_obj_mean_int(all_lbl, C1)
    # C2_all_mean = get_obj_mean_int(all_lbl, C2)
    
    # results = {
    #     "name"        : path.stem,
    #     "label"       : np.arange(1, np.max(lbl) + 1),
    #     "volume_cyt"  : volume_cyt,
    #     "volume_mbn"  : volume_mbn,
    #     "volume_all"  : volume_all,
    #     "C1_cyt_mean" : C1_cyt_mean, "C2_cyt_mean" : C2_cyt_mean,
    #     "C1_mbn_mean" : C1_mbn_mean, "C2_mbn_mean" : C2_mbn_mean,
    #     "C1_all_mean" : C1_all_mean, "C2_all_mean" : C2_all_mean,
    #     "C1_cyt_sum"  : C1_cyt_mean * volume_cyt,
    #     "C2_cyt_sum"  : C2_cyt_mean * volume_cyt,
    #     "C1_mbn_sum"  : C1_mbn_mean * volume_mbn,
    #     "C2_mbn_sum"  : C2_mbn_mean * volume_mbn,
    #     "C1_all_sum"  : C1_all_mean * volume_all,
    #     "C2_all_sum"  : C2_all_mean * volume_all,
    #     "bin_centers_prf" : C1_cyt_mean_prf[:, 0],
    #     "C1_cyt_mean_prf" : C1_cyt_mean_prf[:, 1],
    #     "C2_cyt_mean_prf" : C2_cyt_mean_prf[:, 1],
    #     "C1_mbn_mean_prf" : C1_mbn_mean_prf[:, 1],
    #     "C2_mbn_mean_prf" : C2_mbn_mean_prf[:, 1],
    #     "C1_all_mean_prf" : C1_all_mean_prf[:, 1],
    #     "C2_all_mean_prf" : C2_all_mean_prf[:, 1],
    #     }

    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
        
    # # Save
    # t0 = time.time()
    # print("save    : ", end="", flush=False)
    
    # # pkl
    # with open(out_path / "results.pkl", "wb") as f:
    #     pickle.dump(results, f)
    
    # # csv
    # results_df = pd.DataFrame({k: v for k, v in results.items() if "prf" not in k})   
    # results_prf_df = pd.DataFrame({k: v for k, v in results.items() if "prf" in k})   
    # results_df.to_csv(out_path / "results.csv", index=False)
    # results_prf_df.to_csv(out_path / "results_prf.csv", index=False)
    
    # # tif
    # io.imsave(out_path / "lbl.tif", lbl, check_contrast=False)
    # io.imsave(out_path / "cyt_lbl.tif", cyt_lbl, check_contrast=False)
    # io.imsave(out_path / "mbn_lbl.tif", mbn_lbl, check_contrast=False)
    # io.imsave(out_path / "all_lbl.tif", all_lbl, check_contrast=False)
    # if bsub:
    #     io.imsave(out_path / "C1_bsub.tif", C1, check_contrast=False)
    #     io.imsave(out_path / "C2_bsub.tif", C2, check_contrast=False)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # # Display
    # viewer = napari.Viewer()
    
    # viewer.add_image(
    #     C2, name="C2", visible=1,
    #     colormap="gray", 
    #     gamma=1.0, opacity=1.00,
    #     contrast_limits = parameters["C2_contrast_limits"],
    #     )
    # viewer.add_image(
    #     C1, name="C1", visible=0,
    #     colormap="gray", 
    #     gamma=1.0, opacity=1.00,
    #     contrast_limits = parameters["C1_contrast_limits"],
    #     )
    
    # viewer.add_image(
    #     prd, name="prd", visible=0,
    #     colormap="inferno", blending="additive",  
    #     gamma=1.0, opacity=0.25,
    #     )
    
    # viewer.add_image(
    #     msk0, name="msk0", visible=0,
    #     colormap="bop orange", blending="additive",  
    #     gamma=1.0, opacity=0.25,
    #     rendering="attenuated_mip", attenuation=0.5,
    #     )
    # viewer.add_image(
    #     msk1, name="msk1", visible=0,
    #     colormap="bop blue", blending="additive",  
    #     gamma=1.0, opacity=0.25,
    #     rendering="attenuated_mip", attenuation=0.5,
    #     )
        
    # viewer.add_labels(
    #     cyt_lbl, name="cyt_lbl", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     )
    # viewer.add_labels(
    #     mbn_lbl, name="mbn_lbl", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     )
    # viewer.add_labels(
    #     lbl, name="all_lbl", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     )   
    # viewer.add_labels(
    #     lbl, name="lbl", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     )   
    
    # viewer.add_image(
    #     sdist, name="sdist", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     ) 

#%%

    # # Parameters
    # paths = list(parameters["data_path"].glob("*.nd2"))
    # prf_paths = list(parameters["data_path"].rglob("*_prf.csv"))
    
    # # -------------------------------------------------------------------------
    
    # def filter_data(df, tags):
    #     if tags:
    #         mask = df["name"].apply(lambda x: all(tag in x for tag in tags))
    #     else:
    #         mask = pd.Series(True, index=df.index)
    #     return df.loc[mask]

    # # -------------------------------------------------------------------------

    # # Load, merge, group & save results
    # results_m = []
    # for path in paths:
    #     out_path = path.parent / path.stem
    #     results_m.append(pd.read_csv(out_path / "results.csv"))
    # results_m = pd.concat(results_m, ignore_index=True)
    # results_m_g = results_m.groupby("name").mean(numeric_only=True).reset_index()
    # for i in results_m_g.index:        
    #     results_m_g.loc[i, "label"] = len(
    #         filter_data(results_m, [results_m_g.loc[i, "name"]]))
    # results_m_g = results_m_g.rename(columns={"label": "nlabel"})
    # results_m.to_csv(parameters["data_path"] / "results_m.csv", index=False)
    # results_m_g.to_csv(parameters["data_path"] / "results_m_g.csv", index=False) 
    
    # # Load & merge results_prf
    # results_prf_m = []
    # for path in paths:
    #     out_path = path.parent / path.stem
    #     results_prf_m.append((
    #         path.stem,
    #         pd.read_csv(out_path / "results_prf.csv"),
    #         ))

    # # Plot --------------------------------------------------------------------
    
    # cond = "PEG34"
    # channel = "C1"
    # timepoints = ["00min", "15min", "30min"]
    # fig = plt.figure(figsize=(9, 9), layout="tight")
    # gs = GridSpec(3, 3, figure=fig)
            
    # # Metrics & axis mapping 
    # metrics = [
    #     ("cyt_mean"    , (0, 0)),
    #     ("mbn_mean"    , (1, 0)),
    #     ("all_mean"    , (2, 0)),
    #     ("cyt_sum"     , (0, 1)),
    #     ("mbn_sum"     , (1, 1)),
    #     ("all_sum"     , (2, 1)),
    #     ("cyt_mean_prf", (0, 2)),
    #     ("mbn_mean_prf", (1, 2)),
    #     ("all_mean_prf", (2, 2)),
    #     ]
    
    # # Axes dict.
    # axes = {
    #     metric: fig.add_subplot(gs[row, col]) 
    #     for metric, (row, col) in metrics
    #     }
        
    # # Plot dots + bars
    # for m, (metric, ax) in enumerate(axes.items()):
    #     key = f"{channel}_{metric}"
    #     if m < 6:
    #         for t, tp in enumerate(timepoints):
    #             data = filter_data(results_m_g, [cond, tp])[key]
    #             ax.bar((t,) * len(data), np.mean(data), color="lightgray")
    #             ax.scatter((t,) * len(data), data)
    #         ax.set_title(f"({channel}) {cond}_" + metric)
    #         ax.set_xticks([0, 1, 2])
    #         ax.set_xticklabels(timepoints)     
    
    # # Plot profiles
    # for m, (metric, ax) in enumerate(axes.items()):
    #     key = f"{channel}_{metric}"
    #     bin_centers = np.array(results_prf_m[0][1]["bin_centers_prf"])
    #     if m >= 6:
    #         for t, tp in enumerate(timepoints):
    #             data = []
    #             for r in results_prf_m: 
    #                 if f"{cond}_{tp}" in r[0]:
    #                     data.append(np.array(r[1][key]))
    #             ax.plot(bin_centers, np.mean(np.stack(data), axis=0))
    #         ax.axvline(0, color="k", linestyle="--", linewidth=1)
    #         ax.set_title(f"({channel}) {cond}_" + metric)
    #         ax.set_xticks([0, 25, 50, 75])
    #         ax.set_xlabel("dist. to synaptic plane (pix)")     

