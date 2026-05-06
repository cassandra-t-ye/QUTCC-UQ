# QUTCC🤗: Quantile Uncertainty Training and Conformal Calibration
<!-- <p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2507.14760" alt="arXiv"> <img src="https://img.shields.io/badge/paper-arXiv-red" /> </a>
    <a style="text-decoration:none !important;" href="https://cassandra-t-ye.github.io/projects/QUTCC/" alt="website"> <img src="https://img.shields.io/badge/website-Cornell-yellow" /> </a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"> <img src="https://img.shields.io/badge/license-MIT-blue.svg" /> </a> -->

The official implementation of [QUTCC 🤗: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems] <!-- (https://arxiv.org/abs/2507.14760) -->

<p align="center">
  <img src="teaser.gif" alt="QUTCC Overview" width="1000"/>
</p>
<p align="left">
  <em> During training, QUTCC uses a U-Net architecture with quantile embeddings to learn the full spectrum of quantiles simultaneously. This simultaneous quantile regression approach allows the model to more precisely capture the uncertainty inherent in the inverse problem, compared to previous methods. </em>
</p>

## Setup: 
Dependencies can be installed using
```
conda env create -f environment.yml
source activate qutcc
```

## Quickstart: 
We've included weights checkpoints from Im2Im-Deep and QUTCC for all five imaging tasks in this repo. You can download them [here](https://drive.google.com/file/d/1cX5_UDOYjb_cDIt1IBW7p0RHXuDbXO0m/view?usp=drive_link). Once downloaded, decompress the weights with the following command ```xz -d weights.tar.xz```and put them in the QUTCC_eval folder.

Then go to the provided Jupyter notebook ```quickstart.ipynb``` in the main QUTCC directory and run all code cells. This notebook provides visualization of image samples, as well as for conformalized PDFs. This notebook pulls from ```params.yml```, which contains precomputed lambda, upper quantile, and lower quantile values. **Be sure to select your experiment of interest** before running. 

Additionally, in the ```QUTCC_eval/QUTCC_results``` directory, we've included the individual results, such as the interval length and size-stratified risk from the QUTCC submission. 

## Training + Evaluation:
We break the training and evaluation section into separate parts. 
1. [Data Processing](#data-processing) - Download and preprocess datasets for 5 different inverse imaging tasks
2. [Training](#training) - Train QUTCC models using provided Slurm scripts for each task
3. [Calibration](#calibration) - Calibrate trained models to achieve desired coverage levels
4. [Evaluation](#evaluation) - Evaluate model performance and generate uncertainty metrics
   
## Data Processing:
We evaluate QUTCC on 5 different inverse tasks, which we provide support for below. To train your model for a specific task, please follow the instructions for the task you're interested in. When loading data, make sure that the path ``` /path/to/your/data ``` in the training bash scripts correctly points to your data. Once downloaded and processed, please proceed to the training portion.  

**Available Tasks:**
- [MRI dataset](#fastmri-dataset)
- [Gaussian/Poisson/Real Noise Dataset](#gaussianpoissonreal-noise-dataset)
- [QPI Dataset](#qpi-dataset)

---

### FastMRI dataset
* Download ```knee_singlecoil_train``` from the [FastMRI](https://fastmri.med.nyu.edu/) dataset. 
* In the QUTCC directory, run 
```
python datasets/fastmri/data_processing.py -d [path/to/fast-mri/dataset]
```

### Gaussian/Poisson/Real Noise Dataset
* Download the [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence).
* For the Gaussian or Poisson task, run
```
python datasets/fmd/data_processing.py -d [path/to/fmd/dataset]
```
* For the Real Noise task, run
```
python datasets/fmd/data_processing_realnoise.py -d [path/to/fmd/dataset]
```

### QPI Dataset
* Download the [BSCCM Dataset](https://github.com/Waller-Lab/BSCCM), following the instructions from the repo. Make sure to download the full code repository, not the tiny version. 

## Training:
To train models, use ``train.py`` with the relevant parameters (for learning rate and weight decay, they are already correctly initialized). An example of Gaussian is included below. We recommend running all experiments on GPUs (for all results in the paper, A6000s were used). 

```
python -u train.py \
    --net unet_im2im \
    --transform "center_crop" \
    --epochs 50 \
    --experiment-type "Denoising" \
    --data-root /PATH/TO/DATA/HERE \
    --in-channels 1 \
    --noise-type "gaussian" \
    --sigma 0.5 \
    --exp-name "gaussian" \
    --ckpt-freq 5 \
    --batch-size 4 \
    --plot-freq 1000
```

## Calibration:
To calibrate your model on **one alpha value**, first go to ```analysis.py``` and fill in BEST_RUNS with the experiment of interest, model, and paths to the saved model checkpoints. 
Then go to ```calibration_sweep.py``` and do the following. 
1. Line 123: Choose which experiment you want to calibrate (can do multiple)
2. Line 126: Choose which model you want to calibrate (can do multiple)
3. Line 138: Specify which GPU partition you would like to calibrate on.
Afterwards, run ```python calibration_sweep.py```. This will sweep through all the model checkpoints you have saved and calibrate them to the specified alpha.

The code to produce a **conformalized PDF distribution** can be found in ```evaluation/conformal_pdf_calibration.py```. Before running, be sure to put in the details of your experiment in the lines commented **FILL IN**. The code is currently written to calibrate a gaussian task, but this can be easily switched out for your task of interest. 


## Evaluation
To recreate table 2 (image reconstruction errors) in the paper, use ``calculate_statistics`` in ``analysis.py`` with all model paths being updated. This calculates image reconstruction performances over MSE, SSIM, PSNR, and LPIPS. To recreate table 3 (quantile crossings in $[0.1, 0.2, \dots, 0.9 ]$) in the paper, use ``check_crossings`` with the default quantiles. Both methods are specified in ``analysis.py`` and can be added to construct relevant tables.
