# VesselSemble
An evaluation of top-performing models for accurate, complete liver vessel segmentation. For this study we looked at the models from by the nnUNet and GATSegDiff pipelines for vessel segmentation, and we were specifically looking at how to GATSegDiff model reacts to noisier, less polished data, as well how the model reacts toawrds other models trained and tested on such data when combined together.

![alt text](/VesselSemble/IMG_1522.jpeg)

These are the links to all necessary repositories for setting the necessary pipelines. They both provide ways to build the model from scratch, but they both also have the models available to download as pretraines weights:
- https://github.com/ZhangXiaotong015/GATSegDiff 
- https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1 - For this experiment you would need to download the MSD Task08 HeapicVessel dataset from the Medical Decathlon google drive link. The link is here:
https://drive.google.com/file/d/1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS/view?usp=drivesdk

These are the links to the pretrained weights that we used for the study. The nnUNet model is the one from the Zenodo link, and the GATSegDiff model is the one from the google dirve link:
- https://zenodo.org/records/4003545 - Look for and Dowanload Task008_HepaticVessel.zip.
- https://drive.google.com/drive/folders/1V9NtZingw9XQmFGbGGFOE59vtF6iDeck - Look for and Download LiVS_reannotated_subset_F0_model160000.pt.

Other Prequisites:
- CUDA Toolkit
- Slurm (open source, we used slurm-25.11.2)
- Munge (you can get from pip)
- Python Interpreter - 3.10.0 version (you can also use pyenv) 

Note: Please follow the specficic configuration requirements for each listed GitHub repository. For nnUNet you would need to do all preprocessing steps before you can download the model an run inferences on it. The model I used was the 3D-fullres configuration, and I used the Inference Script that nnUNetv1 provides. For GATSegDiff the only you's need to do is follow the file structure for initiating your pre-trained weights and install the necessary prequisites from their requirements.txt file. There are some other dependencies that will be needed and I have provided them in this repository.

First Setup:
* Inference
    `sbatch Infer_MSD_2.sh`
* Ensemble
     `sbatch Ensemble.sh`

Second Setup:
* Inference
     `sbatch Infer_MSD_3.sh`
* Ensemble
     `sbatch Ensemble_2.sh`

For both ensembles there are BinaryMask processing scripts provided so as to be able to view them in image software. One is meant for use with the ensemble concerning the 3Dircadb1 dataset and another for use with the ensemble concerning the Task08 Hepatic Vessel dataset. They are linked in the repository. 

We used a graph created out of the label annotations of the Liver CT Deformable Image Registration Validation Dataset. The Graph is provided in this repository for use when running analysis on the models, and the script for running performance metrics are provided for use as well. The link to the Github repository connected to that dataset is provided here: https://github.com/deshanyang/Liver-DIR-QA.
