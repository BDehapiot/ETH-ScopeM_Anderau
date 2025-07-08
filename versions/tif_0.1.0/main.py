#%% Imports -------------------------------------------------------------------

import time
import shutil
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed

# bdtools
from bdtools.models.unet import UNet

# functions
from functions import get_tif_data

# Skimage
from skimage.measure import label
from skimage.filters import gaussian
from skimage.morphology import (
    ball, binary_erosion, remove_small_holes,
    )
from skimage.segmentation import (
    clear_border, watershed, expand_labels, relabel_sequential
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

#%% Inputs --------------------------------------------------------------------

# Procedure
procedure = {
    
    "extract" : 0,
    "predict" : 0,
    "process" : 0,
    "analyse" : 0,
    "display" : 1,
    
    }

# Parameters
parameters = {
    
    # Paths
    "data_path" :
        Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Anderau\data"),
    "model_name" :
        "model_256_edt_4000-936_1",
        
    # Process
    
        # Get labels
        "sigma0"  : 0.5,
        "sigma1"  : 2,
        "thresh0" : 0.05,
        "thresh1" : 0.5,
        "remove_border_objects" : False,
    
        # Get intensities
        "dist"    : 3,
        
    # Analyse
    "data"  : "C1_mbn_int",
    "tags0" : ["00min", "PEG12"],
    "tags1" : ["30min", "PEG12"],

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
        self.paths = list(self.data_path.glob("*.tif"))
                
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
            C1 = get_tif_data(path, slc="all", chn=0, rscale=True)
            C2 = get_tif_data(path, slc="all", chn=1, rscale=True)
            C3 = get_tif_data(path, slc="all", chn=2, rscale=True)
            C1 = (C1 // 16).astype("uint8")
            C2 = (C2 // 16).astype("uint8")
            C3 = (C3 // 16).astype("uint8")
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
            # Save
            t0 = time.time()
            print("save : ", end="", flush=False)
            io.imsave(out_path / "C1.tif", C1, check_contrast=False)
            io.imsave(out_path / "C2.tif", C2, check_contrast=False)
            io.imsave(out_path / "C3.tif", C3, check_contrast=False)
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
        
        def _process(path):
            
            out_path = path.parent / path.stem
            lbl_path = out_path / "lbl.tif"
            
            if lbl_path.exists() and self.procedure["process"] == 1:
            
                return
            
            else:

                # Load
                t0 = time.time()
                print(f"{path.stem}")
                print("load : ", end="", flush=False)
                C1 = io.imread(out_path / "C1.tif")
                C2 = io.imread(out_path / "C2.tif")
                prd = io.imread(out_path / "prd.tif")
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
                # Process
                t0 = time.time()
                print("process : ", end="", flush=False)
                
                def remove_small_obj(lbl, min_size=1e4): # parameter
                    vals, counts = np.unique(lbl.ravel(), return_counts=True)
                    msk = (vals != 0) & (counts >= min_size)
                    valid_vals = vals[msk]
                    msk_img = np.isin(lbl, valid_vals)
                    lbl_cleaned = np.where(msk_img, lbl, 0)
                    return relabel_sequential(lbl_cleaned)[0]
                
                def get_obj_area(lbl):
                    vals, counts = np.unique(lbl.ravel(), return_counts=True)
                    return counts[1:]
                                
                def get_obj_int(lbl, img):
                    labels = np.unique(lbl)[1:]
                    return mean(img, labels=lbl, index=labels)
                
                # Get labels
                msk0 = gaussian(prd, sigma=sigma0, preserve_range=True) > thresh0
                msk0 = remove_small_holes(msk0, area_threshold=1e4) # parameter
                if remove_border_objects:
                    msk0 = clear_border(msk0)
                msk1 = gaussian(prd, sigma=sigma1, preserve_range=True) > thresh1
                mrk = label(msk1)
                lbl = watershed(-prd, mrk, mask=msk0).astype("uint8")
                lbl = remove_small_obj(lbl)
                
                # Get intensities
                cyt_msk = binary_erosion(lbl > 0, footprint=ball(dist))
                cyt_lbl = lbl.copy()
                cyt_lbl[~cyt_msk] = 0
                cyt_lbl = cyt_lbl.astype("uint8")
                mbn_lbl = expand_labels(lbl, distance=dist)
                mbn_lbl[cyt_msk] = 0
                mbn_lbl = mbn_lbl.astype("uint8")

                # Get results
                results = {
                    "name"       : path.stem,
                    "label"      : np.arange(1, np.max(lbl) + 1),
                    "area"       : get_obj_area(lbl),
                    "C1_cyt_int" : get_obj_int(cyt_lbl, C1),
                    "C2_cyt_int" : get_obj_int(cyt_lbl, C2),
                    "C1_mbn_int" : get_obj_int(mbn_lbl, C1),
                    "C2_mbn_int" : get_obj_int(mbn_lbl, C2),
                    }
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
                # Save
                t0 = time.time()
                print("save : ", end="", flush=False)
                
                # csv
                results = pd.DataFrame(results)   
                results.to_csv(out_path / "results.csv", index=False)
                
                # tif
                io.imsave(out_path / "lbl.tif", lbl, check_contrast=False)
                io.imsave(out_path / "cyt_lbl.tif", cyt_lbl, check_contrast=False)
                io.imsave(out_path / "mbn_lbl.tif", mbn_lbl, check_contrast=False)
                
                t1 = time.time()
                print(f"{t1 - t0:.3f}s")
                
        # Fetch 
        sigma0  = self.parameters["sigma0"]
        sigma1  = self.parameters["sigma1"]
        thresh0 = self.parameters["thresh0"] * 255
        thresh1 = self.parameters["thresh1"] * 255
        remove_border_objects = self.parameters["remove_border_objects"]
        dist    = self.parameters["dist"]
        
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
        
        def plot(df0, df1):
                        
            # Data
            avgs = [df0[data].mean(), df1[data].mean()]
            errs = [df0[data].std() , df1[data].std() ]
            x = np.arange(2) 
            
            # Plot
            plt.bar(x, avgs, yerr=errs, capsize=20, color="lightgray")
            
            # Format
            plt.title(data)
            plt.xticks(x, [f"{'-'.join(tags0)}", f"{'-'.join(tags1)}"])
            plt.ylabel("fluo. int. (A.U.)")
            
        # Fetch 
        data = self.parameters["data"]
        tags0 = self.parameters["tags0"]
        tags1 = self.parameters["tags1"]
            
        # Load & merge results
        results_m = []
        for path in self.paths:
            out_path = path.parent / path.stem
            results_m.append(pd.read_csv(out_path / "results.csv"))
        results_m = pd.concat(results_m, ignore_index=True)
        
        # Filters results
        results_0 = filter_data(results_m, tags0)
        results_1 = filter_data(results_m, tags1)
        
        # Plot
        plot(results_0, results_1)
        
        # Save
        title0 = f"results_0_{'-'.join(tags0)}.csv"
        title1 = f"results_1_{'-'.join(tags0)}.csv"
        results_m.to_csv(self.data_path / "results_m.csv", index=False)
        results_0.to_csv(self.data_path / title0, index=False)
        results_1.to_csv(self.data_path / title1, index=False)

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
        self.paths = list(self.data_path.glob("*.tif"))
        
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

    # # Imports 
    
    # import matplotlib.pyplot as plt

    # # Parameters
    
    # data = "C1_mbn_int"
    # tags0 = ["00min", "PEG34"]
    # tags1 = ["30min", "PEG34"]
    
    # # -------------------------------------------------------------------------

    # data_path = parameters["data_path"]
    # paths = list(data_path.glob("*.tif"))

    # # -------------------------------------------------------------------------
    
    # results_m = []
    # for path in paths:
    #     out_path = path.parent / path.stem
    #     results_m.append(pd.read_csv(out_path / "results.csv"))
    # results_m = pd.concat(results_m, ignore_index=True)
    
    # # -------------------------------------------------------------------------
    
    # def filter_data(df, tags):
    #     if tags:
    #         mask = df["name"].apply(lambda x: all(tag in x for tag in tags))
    #     else:
    #         mask = pd.Series(True, index=df.index)
    #     return df.loc[mask]

    # df0 = filter_data(results_m, tags0)
    # df1 = filter_data(results_m, tags1)
    
    # # -------------------------------------------------------------------------
        
    # # Data
    # avgs = [df0[data].mean(), df1[data].mean()]
    # sems = [df0[data].sem() , df1[data].sem() ]
    # x = np.arange(2) 
    
    # # Plot
    # plt.bar(
    #     x, avgs, yerr=sems, 
    #     color="lightgray", capsize=10
    #     )
    
    # # Format
    # title = (
    #     f"{data}\n"
    #     f"tags0 : {tags0}\n"
    #     f"tags1 : {tags1}\n"
    #     )
        
    # plt.title(title)
    # plt.xticks(x, ["cond0", "cond1"])
    # plt.ylabel("fluo. int. (A.U.)")
    
#%% 
    
    # htk_idx = 21
    # data_path = parameters["data_path"]
    # paths = list(data_path.glob("*.tif"))
    # path = paths[htk_idx]
    # out_path = path.parent / path.stem
    # C1 = io.imread(out_path / "C1.tif")
    # C2 = io.imread(out_path / "C2.tif")
    # prd = io.imread(out_path / "prd.tif")
    
    # # -------------------------------------------------------------------------
    
    # # Parameters
    # sigma0  = parameters["sigma0"]
    # sigma1  = parameters["sigma1"]
    # thresh0 = parameters["thresh0"] * 255
    # thresh1 = parameters["thresh1"] * 255
    # remove_border_objects = parameters["remove_border_objects"]
    # dist = parameters["dist"]

    # # -------------------------------------------------------------------------

    # def remove_small_obj(lbl, min_size=1e4):
    #     vals, counts = np.unique(lbl.ravel(), return_counts=True)
    #     msk = (vals != 0) & (counts >= min_size)
    #     valid_vals = vals[msk]
    #     msk_img = np.isin(lbl, valid_vals)
    #     lbl_cleaned = np.where(msk_img, lbl, 0)
    #     return relabel_sequential(lbl_cleaned)[0]
    
    # def get_obj_area(lbl):
    #     vals, counts = np.unique(lbl.ravel(), return_counts=True)
    #     return counts[1:]
    
    # def get_obj_int(lbl, img):
    #     labels = np.unique(lbl)[1:]
    #     return mean(img, labels=lbl, index=labels)   

    # t0 = time.time()
    # print("process : ", end="", flush=False)

    # # Get labels
    # msk0 = gaussian(prd, sigma=sigma0, preserve_range=True) > thresh0
    # msk0 = remove_small_holes(msk0, area_threshold=1e4) # parameter
    # if remove_border_objects:
    #     msk0 = clear_border(msk0)
    # msk1 = gaussian(prd, sigma=sigma1, preserve_range=True) > thresh1
    # mrk = label(msk1)
    # lbl = watershed(-prd, mrk, mask=msk0)
    # lbl = remove_small_obj(lbl)
 
    # # Get intensities
    # cyt_msk = binary_erosion(lbl > 0, footprint=ball(dist))
    # mbn_lbl = expand_labels(lbl, distance=dist)
    # mbn_lbl[cyt_msk] = 0
    # cyt_lbl = lbl.copy()
    # cyt_lbl[~cyt_msk] = 0
    
    # # Get results
    # results = {
    #     "name"       : path.stem,
    #     "label"      : np.arange(1, np.max(lbl) + 1),
    #     "area"       : get_obj_area(lbl),
    #     "C1_cyt_int" : get_obj_int(cyt_lbl, C1),
    #     "C2_cyt_int" : get_obj_int(cyt_lbl, C2),
    #     "C1_mbn_int" : get_obj_int(mbn_lbl, C1),
    #     "C2_mbn_int" : get_obj_int(mbn_lbl, C2),
    #     }
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
        
    # # -------------------------------------------------------------------------
    
    # t0 = time.time()
    # print("run : ", end="", flush=False)
    
    # mbn_out = mbn_lbl > 0
    # mbn_out = mbn_out ^ binary_erosion(mbn_out)
    
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s")
    
    # # -------------------------------------------------------------------------
    
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
    #     lbl, name="lbl", visible=0,
    #     blending="additive", 
    #     opacity=0.50,
    #     )   
    
    # viewer.add_image(
    #     mbn_out, name="mbn_out", visible=1,
    #     colormap="grey", blending="additive",  
    #     gamma=1.0, opacity=1.00,
    #     )
