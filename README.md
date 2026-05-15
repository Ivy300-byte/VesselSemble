# VesselSemble
The complete evaluation of top-performing models' streangths and weaknesses for accurate, complete liver vessel segmentation.

Links to All Necessary Repositories from other GitHub Creators:
- https://github.com/ZhangXiaotong015/GATSegDiff 
- https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

Links to Other Necessary Folders:
- https://zenodo.org/records/4003545 - Look for and Dowanload Task008_HepaticVessel.zip.
- https://drive.google.com/drive/folders/1V9NtZingw9XQmFGbGGFOE59vtF6iDeck - Look for and Download LiVS_reannotated_subset_F0_model160000.pt.

Other Prequisites:
- CUDA Toolkit
- Slurm (open source, I used slurm-25.11.2)
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
