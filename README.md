# Self Supervised Learning of Optical Flow from Camera Images
 **PyTorch** implementations of the [UnFlow](https://arxiv.org/abs/1711.07837) and [PWC-Net](https://arxiv.org/abs/1709.02371) architectures for self supervised learning of optical flow put together with various loss functions in the recent literature. This work is a part of my Student Research Project (Studienarbeit in German) at TU Brausnchweig.
 - [Report](https://www.dropbox.com/s/lvaspcfctch56d2/Report_Studienarbeit.pdf?dl=0) for the project
 - [Slides](https://www.dropbox.com/scl/fi/i1ccevxyr6kb7nutibshl/Self-Supervised-Learning-of-Optical-Flow.pptx?dl=0&rlkey=2ka00njicngya8w2gyf3twk3g) of the final presentation
 
## Requirements
The following packages need to be installed for execution of the training, evaluation and inference scripts.
- PyTorch 1.3
- Tensorboard
- scikit-image
- PyPNG
- CuPy

## Structure of the Project Directory

```
1. Base Directory of the Project
   - src 
     This directory contains implementations of the network architectures, unsupervised losses, correlation layer, evaluation and inference scripts
   - DataLOaders
     Holds DataLoaders specific to each dataset
   - log
     Directory for storing the training logs, intermediate checkpoints during training. The training runs are separated by name of experiments.
     - name_of_the_experiment_1
       - train (contains event files to visualise training errors in tensorboard)
       - eval (contains event files to visualise validation errors during training) 
       - Intermediate_CKPTS (directory to store checkpoints at specific intervals during training)
       - Final_CKPTs (directory to save the last checkpoint at the end of training)
     - name_pf_the_experiment_2
     - name_of_the_experiment_3
```

## Inference

The inference script could be executed as follows
```
cd src/core/
python3 inference.py --experiment C_KITTI 
                     --arch C 
                     --img_1_path ./sample_images/frame_1.png 
                     --img_2_path ./sample_images/frame_2.png 
                     --save_idr ./
```

- **experiment**:  ``` name_of_experiment``` from which to load the checkpoint for evaluation. For example  ```C_KITTI ```. The checkpoint from directory ``` log/name_of_experiment/Final_CKPTs/``` corresponding to latest __step index__ would be used for estimating the flow.  
- **arch**: ```C``` or ```CS``` or ```CSS```
- **img_1_path**: Path to the first frame relative to the base directory of the project.
- **img_2_path**: Path to the second frame relative to the base directory of the project.
- **save_dir**: Path to the directory relative to the base directory of the project for saving the flow output and it's color code visualisation. If the specified directory doesn't exist, it will be created. Two files will be written: 
  - **flow.npy**: Optical flow output stored as an numpy array of soze **H x W x 2**
  - **flow_vis.png**: Color coded RGB image of optical flow visualisation. 
  
## Evaluation

The evaluation script evaluates the performance of trained models on the dataset with flow ground truths, in our case, the optical flow datasets of [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow) and [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow). It could be executed as follows
```
cd src/core/
python3 eval.py --experiment C_KITTI 
                --arch C 
                --dataset KITTI_2015 
                --dataset_dir  /path/to/dataset/dir 
                --save_dir ./C_KITTI_2015
```

- **experiment**:  ``` name_of_experiment``` from which to load the checkpoint for evaluation. For example  ```C_KITTI ```. The checkpoint from directory ``` log/name_of_experiment/Final_CKPTs/``` corresponding to latest __step index__ would be used for estimating the flow.  
- **arch**: ```C``` or ```CS``` or ```CSS```
- **dataset**: Dataset to evaluate evaluate on, either ```KITTI_2012``` or ```KITTI_2015``` 
- **dataset_dir**: Path to the directory of the evaluation dataset relative to the base directory of the project. For example (in the cluster) 
``` /path/to/shared/folder/kitti_flow_2012/KITTI_Flow_2012/training```
- **save_dir**: Path to the directory relative to the path of the base directory to save the results. The directory will be created, if not exists. Following sub_directories will be created
  - **flow**: Will contain the optical flow outputs in __.npy__ format.
  - **flow_vis**: Will contain the color coded visualisations of optical flow outputs.
  - **flow_gt_vis**: Will contain the color coded visualizations of ground truth optical flow.
  - **occ**: Will contain the occlusion maps estimated using optical flow outputs from network.
  - **occ_gt**: Will contain the occlusion maps estimated using ground truth flow.
  - **flow_error**: Will contain the flow error maps estimated by comparing predicted flow with ground truth flow.
  
## Training

First set the options corresponding to training in **config.ini** file. 
Then execute the training script **run.py** as follows

```
python3 run.py --config_path ./config.ini 
               --experiment name_of_experiment 
               --ckpt_filename ckpt 
               --ow False
```

- **config_path**: path to **config.ini**. (Dont change the default location, which is ```./config.ini```)
- **experiment**: name of the experiment for the training session. A directory ```log/name_of_experiment/```will be created for the corresponding experiment 
- **ckpt_filename**: The name of the files to save the ckeckpoints with. This name is appended by the **global step index** corresponding to the state at which the checkpoint is saved during training. For example, if the option is set as ```ckpt```, then the checkpoints will be saved as
```ckpt_500.pytorch, ckpt_1000.pytorch, ckpt_1500.pytorch and so on```.
- **ow**: Whether to owerwrite the specified **experiment**, if already exists. If set to **True** and if the experiment already exists, the directory corresponding to the specified experiment will be erased and training will restart from **step 0**. If **False**, and if the specified experiment already exists, it will resume the training from the last saved checkpoint state.
