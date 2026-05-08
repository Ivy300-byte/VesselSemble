# VesselSemble
The complete evaluation of top-performing models' streangths and weaknesses for accurate, complete liver vessel segmentation.

Links to All Necessary Repositories from other GitHub Creators:
- https://github.com/ZhangXiaotong015/GATSegDiff 
- https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1

Links to Other Necessary Folders:
- https://zenodo.org/records/4003545 - Look for and Dowanload Task008_HepaticVessel.zip.
- https://drive.google.com/drive/folders/1V9NtZingw9XQmFGbGGFOE59vtF6iDeck - Look for and Download LiVS_reannotated_subset_F0_model160000.pt.

Other things you will need:
- CUDA Toolkit
- Slurm
- Python Interperter - 3.10.0 version (you can also use pyenv) 

Note: Please follow the specficic configuration requirements for each listed GitHub repository. For nnUNet you would need to do all preprocessing steps before you can download the model an run inferences on it. The model I used was the 3D-fullres configuration. For GAT the only you's need to do is follow the file structure for initiating you pre-trained weights.




