# 3DVNet: Multi-View Depth Prediction and Volumetric Refinement
## [Project Page](https://alexrich021.github.io/3dvnet/) | [Paper](https://arxiv.org/abs/2112.00202) | [Weights](https://drive.google.com/drive/folders/1RP7TSVYhQQbANygJuxwbLHHuLvBsVXa6?usp=sharing) | [Supplementary](https://alexrich021.github.io/3dvnet/static/3dvnet_3dv2021_supplementary.pdf)

## Dependencies/Installation
The following will install all dependencies for 3DVNet training and evaluation. This installation has only been tested on Ubuntu 20.04.
```
conda create -n 3dvnet python=3.8 -y
conda activate 3dvnet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install \
  pytorch-lightning==1.1.2 \
  wandb \
  tqdm \
  opencv-python \
  open3d==0.11.2 \
  scikit-image==0.17.2 \
  pyrender \
  trimesh \
  kornia==0.4.1 \
  path

# PyTorch Geometric installation
pip install torch-scatter==2.0.5 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-cluster==1.5.8 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.6.3

# Minkowski Engine installation (https://github.com/NVIDIA/MinkowskiEngine)
# note our code is only tested with Minkowski Engine 0.5.0
conda install openblas-devel -c anaconda -y
export CUDA_HOME=/usr/local/cuda-11.xx   # replace with local cuda version >=11.0
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# finally, install code as editable local package
python -m pip install --editable . --user
```
The following will install all additional dependencies for evaluating competing baselines. Should any dependency issues arise, we refer you to the author's original codebase for specific installation instructions. 
```
# Atlas (note installation has been changed from original Atlas to match our cudatoolkit/pytorch version)
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

# NeuralRecon (torchsparse: https://github.com/mit-han-lab/torchsparse)
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.2.0
pip install loguru transforms3d
```

## Data
See `data_preprocess` directory for data preprocessing scripts for ScanNet, ICL-NUIM, and TUM-RGBD. We briefly outline the ScanNet preprocessing.

Given an existing [ScanNet](https://github.com/ScanNet/ScanNet) directory `scannet_src` extracted using the author provided [extraction scripts](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python), run
```
python data_preprocess/preprocess_scannet.py --src path/to/scannet_src --dst path/to/new/scannet_dst
```
**NOTE: If `--src` and `--dst` are the same directory, you will overwrite your existing ScanNet dataset.**

This script expects a ScanNet src directory with structure
```
scannet_src/
    scannetv2_*.txt
    scans*/
        scene_****_**/
            scene_****_**_vh_clean_2.ply
            color/
            depth/
            intrinsic/
            pose/
```
and creates a new ScanNet directory with structure
```
scannet_dst/
    scannetv2_*.txt
    scans*/
        scene_****_**/
            info.json
            scene_****_**_vh_clean_2.ply
            color/
            depth/
```
with depth and color images resized properly. The `info.json` file is used by the dataloader for frame selection. Once data has been preprocessed, modify `mv3d/config.py` such that `SCANNET_DIR = path/to/scannet_dst`. `mv3d/dsets/dataset.py` can be run as a script to visualize a random ScanNet scene. This can be used for convenient debugging of the data preprocessing.

#### A note on our data batch setup
We use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) edge indexing to specify our reference and source images in each batch from our dataloader. Specifically, our batch of images is returned from the dataloader flattened, i.e. `batch.images.shape = [num_images, 3, 256, 320]`. The attribute `batch.ref_src_idx` of shape `[2, num_source_images]` then specifies the corresponding reference and source images, and `batch.images_batch` specifies the corresponding batch index of each image. We find the flexibility of this setup quite convenient. See the linked PyTorch Geometric documentation for more details.

## Training
All training parameters can be changed in `mv3d/config.py`. For logging, we use [wandb](https://wandb.ai/). It should be quite easy to modify the code to use TensorBoard. Initial training is done with the CNN backbone fixed. Simply run `python train.py`. No args are used, all parameters are specified in `mv3d/config.py`. Once a model has converged in the previous run (~120 epochs), specify model save location using the `PATH` variable in `mv3d/config.py`. Run `mv3d/finetune.py` until convergence (~100 epochs).

For each baseline used for comparison and not trained on ScanNet, we include a fine-tuning script `finetune.py`. We also provide our fine-tuned weights trained using the fine-tuning script.

## Evaluation
To run 3DVNet evaluation, use `mv3d/eval-3dvnet.py`. To run evaluation on each baseline method, use the `mv3d/baselines/${method}/eval-${method}.py`. The evaluation scripts have no arguments. Instead, evaluation configuration can be changed in `mv3d/eval/config.py`. The common code used for evaluation of all methods can be found in `mv3d/eval`. We provide all weights for 3DVNet and all baselines (both author provided and ScanNet finetuned when applicable) [here](https://drive.google.com/drive/folders/1RP7TSVYhQQbANygJuxwbLHHuLvBsVXa6?usp=sharing). To use, simply download and place each `*_weights` directory in the same directory as the corresponding `eval_*.py` script.

#### Point cloud fusion implementation
We use a depth-based multi-view consistency check in point cloud fusion. Code modified from the MVSNet point cloud fusion implementation can be found [here](https://github.com/alexrich021/fusibile). Once built, modify the `FUSIBILE_EXE_PATH` variable in `mv3d/eval/config.py` to point to the built binary. We also provide a **very** slow PyTorch implementation of point cloud fusion. If no fusibile binary is found, our evaluation pipeline defaults to this implementation.

Disparity-based point cloud fusion can also be used. To do this, use the [original MVSNet implementation](https://github.com/yoyo000/fusibile) and modify the `FUSIBILE_EXE_PATH` variable in `mv3d/eval/config.py` to point to the built binary. Note that the `Z_THRESH` variable in `mv3d/eval/config.py` must be updated to a value appropriate for disparity thresholding. We recommend `Z_THRESH=0.125` based on validation results.

#### Results folder structure
The parent folder for evaluation results for all methods is specified in the `SAVE_DIR` variable in `eval/config.py`. When running the evaluation script for a given method, the folder `SAVE_DIR/methodname` will be created with the following structure:
```
methodname/
    metrics_*.json
    scenes/
        scene1/
            preds.npz
            metrics_*.json
            *.ply
        scene2/
            preds.npz
            metrics_*.json
            *.ply
        ...
```
Depth predictions for each scene are stored in `preds.npz` file. The metric filenames are reflective of the metrics they calculate. For example, `metrics_2d.json` contains 2D depth metrics, while `metrics_3d_0.010_3v_masked.json` contains 3D metrics calculated using point cloud fusion with 3-view consistency, a depth consistency threshold of 0.01, and holes in observed regions masked out. The ply files in each scene folder are also correspondingly named. The top-level metrics files contain aggregated metrics.

#### Visualization
We include a convenient Open3D visualization script `mv3d/eval/visualization.py` for visualizing the reconstructions produced by all baseline methods. To change which methods are visualized, modify the 3 lists found at the start of the script. The key callbacks are defined at the end of the script.

#### Evaluating your own data
To evaluate all methods on your own data, you must prepare an `info.json` file for each scene. See the data preprocessing scripts for examples of how to do this. Then, modify `mv3d/eval/config.py` to specify a parent directory for the results. Finally, modify `mv3d/eval/main.py` to properly load the list of custom scenes you wish to evaluate. You can check if your `info.json` file is prepared properly by modifying and running the visualization script at the bottom of `mv3d/dsets/dataset.py`.

#### Evaluating your own method
All of the evaluation scripts follow a common boilerplate structure. Use one of the eval scripts as a starting template. For volumetric methods (or any method that directly predicts 3D scene geometry), `mv3d/baselines/atlas/eval-atlas.py` is the best starting point. For depth-based methods, `mv3d/baselines/gpmvs/eval-gpmvs.py` or `mv3d/baselines/pointmvsnet/eval-pointmvsnet.py` are the best starting points. If your method includes confidence maps for each depth map, use the latter. 

#### Evaluating your own frame selection method
Frame selection classes are found in `mv3d/dsets/frameselector.py`. We provide several options. To write your own, subclass the `FrameSelector` class. To use this in evaluation, modify `mv3d/eval/main.py` to use your new frame selection class. To use this during training or finetuning, modify the corresponding `train.py` or `finetune.py` file where the frame selector is specified. If you intend to only use your frame selection for evaluation, the `seed_idx` argument can be ignored in your subclass.

## Citation
```
@inproceedings{rich20213dvnet,
  title={{3DVNet}: Multi-View Depth Prediction and Volumetric Refinement},
  author={Alexander Rich and Noah Stier and Pradeep Sen and Tobias H\"ollerer},
  booktitle={Proceedings of the International Conference on {3D} Vision (3DV)},
  year={2021}
}
```
