# VesselSemble
An evaluation of top-performing models for accurate, complete liver vessel segmentation. For this study we looked at the models from by the nnUNet and GATSegDiff pipelines for vessel segmentation, and we were specifically looking at how to GATSegDiff model reacts to noisier, less polished data, as well how the model reacts toawrds other models trained and tested on such data when combined together.

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

Some of our datases didn't come wth gorund truth segmentations and we wanted to keeep a uniform mode for data analysis on our models. To tackle this issue, we decided to opt for  unsupervied Topological Data Analysis on all of our predictions, calculating mean_tortuosity, average_branch_length, total_vessel_length, bifurcation_count, fragmentation_errors, and false_loops:  

The results of the nnUNet predictions on the MSD Tak08 Hepatic Vessel Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/nnUNet-on-MSD.csv)

The results of the GATSegDiff on the MSD Tak08 Hepatic Vessel Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/GATSegDiff-on-MSD.csv)

The results of the nUNet predictions on the original 3D-ircadb-01 Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/nnUNet-on-3Dircadb.csv)

The results of the GATSegDiff predictions on the original 3D-ircadb-01 Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/GATSegDiff-on-3Dircadb.csv)

The results of the ensemble of the GATSegDiff and nnUNet predictions on the MSD Task08 HepaticVessel Dataset.

| case_id                             	| model_architecture 	| status  	| total_vessel_length_mm 	| bifurcation_count 	| mean_tortuosity 	| average_branch_length_mm 	| fragmentation_errors_h0 	| false_loops_h1 	|
|-------------------------------------	|--------------------	|---------	|------------------------	|-------------------	|-----------------	|--------------------------	|-------------------------	|----------------	|
| FINAL_ENSEMBLE_hepaticvessel_252    	| nnunet+GATSegDiff  	| Success 	| 185.13                 	| 0                 	| 1.001           	| 185.13                   	| 0                       	| 0              	|
| FINAL_VESSEL_ONLY_hepaticvessel_003 	| nnunet+GATSegDiff  	| Success 	| 1196.59                	| 24                	| 1.264           	| 21.76                    	| 180                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_009 	| nnunet+GATSegDiff  	| Success 	| 1433.45                	| 25                	| 1.402           	| 21.39                    	| 192                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_012 	| nnunet+GATSegDiff  	| Success 	| 1125.22                	| 19                	| 1.279           	| 20.46                    	| 175                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_015 	| nnunet+GATSegDiff  	| Success 	| 1264.59                	| 18                	| 1.331           	| 28.1                     	| 169                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_017 	| nnunet+GATSegDiff  	| Success 	| 1515.35                	| 29                	| 1.388           	| 21.65                    	| 188                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_021 	| nnunet+GATSegDiff  	| Success 	| 2081.05                	| 37                	| 1.289           	| 22.14                    	| 216                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_014 	| nnunet+GATSegDiff  	| Success 	| 1199.36                	| 25                	| 1.207           	| 20.33                    	| 173                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_034 	| nnunet+GATSegDiff  	| Success 	| 972.72                 	| 24                	| 1.254           	| 16.21                    	| 134                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_035 	| nnunet+GATSegDiff  	| Success 	| 1301.54                	| 22                	| 1.301           	| 26.03                    	| 162                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_036 	| nnunet+GATSegDiff  	| Success 	| 1397.3                 	| 26                	| 1.283           	| 21.5                     	| 171                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_038 	| nnunet+GATSegDiff  	| Success 	| 1411.87                	| 33                	| 1.405           	| 19.08                    	| 172                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_041 	| nnunet+GATSegDiff  	| Success 	| 1134.36                	| 24                	| 1.33            	| 20.62                    	| 176                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_024 	| nnunet+GATSegDiff  	| Success 	| 848.72                 	| 14                	| 1.369           	| 22.94                    	| 146                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_037 	| nnunet+GATSegDiff  	| Success 	| 1834.65                	| 37                	| 1.209           	| 23.22                    	| 211                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_043 	| nnunet+GATSegDiff  	| Success 	| 1405.88                	| 30                	| 1.305           	| 21.3                     	| 196                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_045 	| nnunet+GATSegDiff  	| Success 	| 1367.28                	| 23                	| 1.299           	| 25.32                    	| 180                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_047 	| nnunet+GATSegDiff  	| Success 	| 1496.87                	| 33                	| 1.266           	| 18.95                    	| 176                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_055 	| nnunet+GATSegDiff  	| Success 	| 1432.15                	| 31                	| 1.489           	| 20.17                    	| 189                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_054 	| nnunet+GATSegDiff  	| Success 	| 442.06                 	| 9                 	| 1.582           	| 22.1                     	| 1                       	| 2              	|
| FINAL_VESSEL_ONLY_hepaticvessel_048 	| nnunet+GATSegDiff  	| Success 	| 1140.2                 	| 38                	| 1.309           	| 15.41                    	| 178                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_060 	| nnunet+GATSegDiff  	| Success 	| 1233.42                	| 23                	| 1.296           	| 22.03                    	| 186                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_056 	| nnunet+GATSegDiff  	| Success 	| 1133.93                	| 16                	| 1.282           	| 22.23                    	| 196                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_063 	| nnunet+GATSegDiff  	| Success 	| 1457.23                	| 36                	| 1.272           	| 16.94                    	| 205                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_069 	| nnunet+GATSegDiff  	| Success 	| 1284.23                	| 25                	| 1.295           	| 21.4                     	| 180                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_073 	| nnunet+GATSegDiff  	| Success 	| 1269.15                	| 25                	| 1.807           	| 22.27                    	| 181                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_074 	| nnunet+GATSegDiff  	| Success 	| 969.79                 	| 17                	| 1.299           	| 24.87                    	| 138                     	| 4              	|
| FINAL_VESSEL_ONLY_hepaticvessel_064 	| nnunet+GATSegDiff  	| Success 	| 1221.74                	| 19                	| 1.255           	| 23.5                     	| 155                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_076 	| nnunet+GATSegDiff  	| Success 	| 1536.76                	| 37                	| 1.275           	| 17.27                    	| 187                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_101 	| nnunet+GATSegDiff  	| Success 	| 1605.52                	| 39                	| 1.306           	| 17.64                    	| 186                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_095 	| nnunet+GATSegDiff  	| Success 	| 1804.66                	| 35                	| 1.364           	| 19.83                    	| 236                     	| 15             	|
| FINAL_VESSEL_ONLY_hepaticvessel_099 	| nnunet+GATSegDiff  	| Success 	| 2266.88                	| 49                	| 1.229           	| 19.38                    	| 210                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_105 	| nnunet+GATSegDiff  	| Success 	| 1620.95                	| 42                	| 1.279           	| 18.01                    	| 197                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_107 	| nnunet+GATSegDiff  	| Success 	| 1784.48                	| 32                	| 1.297           	| 22.31                    	| 222                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_109 	| nnunet+GATSegDiff  	| Success 	| 1924.57                	| 44                	| 1.243           	| 19.25                    	| 201                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_106 	| nnunet+GATSegDiff  	| Success 	| 1275.78                	| 27                	| 1.385           	| 19.33                    	| 153                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_118 	| nnunet+GATSegDiff  	| Success 	| 1395                   	| 27                	| 1.254           	| 24.47                    	| 179                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_120 	| nnunet+GATSegDiff  	| Success 	| 907.44                 	| 13                	| 1.271           	| 26.69                    	| 116                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_122 	| nnunet+GATSegDiff  	| Success 	| 1962.9                 	| 47                	| 1.194           	| 18.34                    	| 208                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_130 	| nnunet+GATSegDiff  	| Success 	| 1643.66                	| 25                	| 1.309           	| 25.68                    	| 179                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_114 	| nnunet+GATSegDiff  	| Success 	| 2076.62                	| 58                	| 1.213           	| 16.61                    	| 241                     	| 14             	|
| FINAL_VESSEL_ONLY_hepaticvessel_126 	| nnunet+GATSegDiff  	| Success 	| 856.71                 	| 15                	| 1.271           	| 21.97                    	| 127                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_134 	| nnunet+GATSegDiff  	| Success 	| 1008.01                	| 21                	| 1.26            	| 19.02                    	| 167                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_135 	| nnunet+GATSegDiff  	| Success 	| 990.65                 	| 15                	| 1.239           	| 24.77                    	| 110                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_143 	| nnunet+GATSegDiff  	| Success 	| 2093.04                	| 47                	| 1.393           	| 19.93                    	| 228                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_149 	| nnunet+GATSegDiff  	| Success 	| 939.54                 	| 16                	| 1.331           	| 24.09                    	| 103                     	| 2              	|
| FINAL_VESSEL_ONLY_hepaticvessel_156 	| nnunet+GATSegDiff  	| Success 	| 1459.54                	| 31                	| 1.269           	| 17.8                     	| 201                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_153 	| nnunet+GATSegDiff  	| Success 	| 1124.85                	| 17                	| 1.28            	| 24.45                    	| 155                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_162 	| nnunet+GATSegDiff  	| Success 	| 1670.65                	| 31                	| 1.394           	| 21.15                    	| 215                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_168 	| nnunet+GATSegDiff  	| Success 	| 1961.19                	| 42                	| 1.304           	| 19.42                    	| 220                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_170 	| nnunet+GATSegDiff  	| Success 	| 1358.97                	| 25                	| 1.24            	| 19.14                    	| 167                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_180 	| nnunet+GATSegDiff  	| Success 	| 1489.15                	| 32                	| 1.307           	| 17.94                    	| 203                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_182 	| nnunet+GATSegDiff  	| Success 	| 1777.73                	| 39                	| 1.285           	| 19.32                    	| 214                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_187 	| nnunet+GATSegDiff  	| Success 	| 2892.59                	| 56                	| 1.228           	| 23.91                    	| 227                     	| 14             	|
| FINAL_VESSEL_ONLY_hepaticvessel_188 	| nnunet+GATSegDiff  	| Success 	| 2039.32                	| 42                	| 1.324           	| 21.93                    	| 221                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_176 	| nnunet+GATSegDiff  	| Success 	| 2554.17                	| 71                	| 1.171           	| 17.26                    	| 247                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_202 	| nnunet+GATSegDiff  	| Success 	| 422.89                 	| 7                 	| 1.264           	| 23.49                    	| 2                       	| 0              	|
| FINAL_VESSEL_ONLY_hepaticvessel_191 	| nnunet+GATSegDiff  	| Success 	| 1455.76                	| 36                	| 1.27            	| 19.41                    	| 163                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_169 	| nnunet+GATSegDiff  	| Success 	| 2181.88                	| 48                	| 1.195           	| 21.18                    	| 234                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_204 	| nnunet+GATSegDiff  	| Success 	| 1334.87                	| 33                	| 1.314           	| 16.9                     	| 179                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_205 	| nnunet+GATSegDiff  	| Success 	| 1794.45                	| 34                	| 1.26            	| 22.43                    	| 182                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_212 	| nnunet+GATSegDiff  	| Success 	| 1586.21                	| 28                	| 1.365           	| 22.34                    	| 195                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_216 	| nnunet+GATSegDiff  	| Success 	| 2562.75                	| 52                	| 1.336           	| 21.36                    	| 237                     	| 19             	|
| FINAL_VESSEL_ONLY_hepaticvessel_220 	| nnunet+GATSegDiff  	| Success 	| 2526.15                	| 53                	| 1.25            	| 19.58                    	| 231                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_228 	| nnunet+GATSegDiff  	| Success 	| 1655.77                	| 31                	| 1.229           	| 23.65                    	| 174                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_190 	| nnunet+GATSegDiff  	| Success 	| 2903.91                	| 75                	| 1.193           	| 18.04                    	| 261                     	| 20             	|
| FINAL_VESSEL_ONLY_hepaticvessel_243 	| nnunet+GATSegDiff  	| Success 	| 856.1                  	| 16                	| 1.193           	| 19.46                    	| 137                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_232 	| nnunet+GATSegDiff  	| Success 	| 2242.97                	| 45                	| 1.252           	| 21.78                    	| 224                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_247 	| nnunet+GATSegDiff  	| Success 	| 1512.33                	| 30                	| 1.307           	| 19.64                    	| 194                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_241 	| nnunet+GATSegDiff  	| Success 	| 2102.95                	| 43                	| 1.289           	| 22.37                    	| 238                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_252 	| nnunet+GATSegDiff  	| Success 	| 1627.47                	| 32                	| 1.25            	| 20.09                    	| 190                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_254 	| nnunet+GATSegDiff  	| Success 	| 1575.82                	| 41                	| 1.237           	| 17.32                    	| 178                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_261 	| nnunet+GATSegDiff  	| Success 	| 1734.26                	| 31                	| 1.444           	| 23.76                    	| 205                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_251 	| nnunet+GATSegDiff  	| Success 	| 1919.94                	| 35                	| 1.166           	| 22.59                    	| 231                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_250 	| nnunet+GATSegDiff  	| Success 	| 2122.34                	| 48                	| 1.202           	| 20.02                    	| 245                     	| 14             	|
| FINAL_VESSEL_ONLY_hepaticvessel_273 	| nnunet+GATSegDiff  	| Success 	| 1950.65                	| 44                	| 1.263           	| 20.97                    	| 194                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_276 	| nnunet+GATSegDiff  	| Success 	| 1838.06                	| 32                	| 1.253           	| 22.98                    	| 206                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_277 	| nnunet+GATSegDiff  	| Success 	| 987.88                 	| 18                	| 1.266           	| 25.33                    	| 106                     	| 4              	|
| FINAL_VESSEL_ONLY_hepaticvessel_298 	| nnunet+GATSegDiff  	| Success 	| 1218.29                	| 24                	| 1.285           	| 19.04                    	| 151                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_267 	| nnunet+GATSegDiff  	| Success 	| 1291.07                	| 27                	| 1.193           	| 19.86                    	| 201                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_302 	| nnunet+GATSegDiff  	| Success 	| 1526.09                	| 27                	| 1.302           	| 21.8                     	| 206                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_264 	| nnunet+GATSegDiff  	| Success 	| 1766                   	| 46                	| 1.198           	| 17.49                    	| 231                     	| 25             	|
| FINAL_VESSEL_ONLY_hepaticvessel_301 	| nnunet+GATSegDiff  	| Success 	| 2311.5                 	| 48                	| 1.24            	| 19.76                    	| 220                     	| 15             	|
| FINAL_VESSEL_ONLY_hepaticvessel_306 	| nnunet+GATSegDiff  	| Success 	| 1532.68                	| 40                	| 1.583           	| 17.03                    	| 185                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_292 	| nnunet+GATSegDiff  	| Success 	| 1716.54                	| 31                	| 1.186           	| 22.59                    	| 201                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_310 	| nnunet+GATSegDiff  	| Success 	| 2025.36                	| 42                	| 1.301           	| 21.1                     	| 211                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_304 	| nnunet+GATSegDiff  	| Success 	| 2778.84                	| 74                	| 1.179           	| 16.94                    	| 255                     	| 21             	|
| FINAL_VESSEL_ONLY_hepaticvessel_312 	| nnunet+GATSegDiff  	| Success 	| 2099.23                	| 42                	| 1.2             	| 19.99                    	| 246                     	| 21             	|
| FINAL_VESSEL_ONLY_hepaticvessel_315 	| nnunet+GATSegDiff  	| Success 	| 2384.51                	| 60                	| 1.21            	| 18.78                    	| 251                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_316 	| nnunet+GATSegDiff  	| Success 	| 2019.26                	| 52                	| 1.241           	| 18.53                    	| 208                     	| 14             	|
| FINAL_VESSEL_ONLY_hepaticvessel_313 	| nnunet+GATSegDiff  	| Success 	| 1694.04                	| 29                	| 1.205           	| 24.91                    	| 231                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_317 	| nnunet+GATSegDiff  	| Success 	| 1738.91                	| 39                	| 1.171           	| 19.32                    	| 224                     	| 12             	|
| FINAL_VESSEL_ONLY_hepaticvessel_331 	| nnunet+GATSegDiff  	| Success 	| 2269.36                	| 50                	| 1.173           	| 19.91                    	| 237                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_328 	| nnunet+GATSegDiff  	| Success 	| 2602.33                	| 55                	| 1.22            	| 21.33                    	| 239                     	| 26             	|
| FINAL_VESSEL_ONLY_hepaticvessel_326 	| nnunet+GATSegDiff  	| Success 	| 2530.68                	| 59                	| 1.195           	| 18.21                    	| 256                     	| 27             	|
| FINAL_VESSEL_ONLY_hepaticvessel_337 	| nnunet+GATSegDiff  	| Success 	| 1659.14                	| 34                	| 1.265           	| 18.85                    	| 209                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_339 	| nnunet+GATSegDiff  	| Success 	| 2289.9                 	| 63                	| 1.188           	| 17.09                    	| 239                     	| 26             	|
| FINAL_VESSEL_ONLY_hepaticvessel_342 	| nnunet+GATSegDiff  	| Success 	| 2013.07                	| 53                	| 1.169           	| 17.35                    	| 234                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_347 	| nnunet+GATSegDiff  	| Success 	| 1530.5                 	| 34                	| 1.438           	| 18.66                    	| 153                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_346 	| nnunet+GATSegDiff  	| Success 	| 2763.43                	| 65                	| 1.243           	| 18.8                     	| 247                     	| 15             	|
| FINAL_VESSEL_ONLY_hepaticvessel_343 	| nnunet+GATSegDiff  	| Success 	| 2928.3                 	| 76                	| 1.207           	| 16.73                    	| 253                     	| 22             	|
| FINAL_VESSEL_ONLY_hepaticvessel_334 	| nnunet+GATSegDiff  	| Success 	| 1110.55                	| 27                	| 1.175           	| 18.21                    	| 192                     	| 4              	|
| FINAL_VESSEL_ONLY_hepaticvessel_351 	| nnunet+GATSegDiff  	| Success 	| 2349.22                	| 58                	| 1.218           	| 16.43                    	| 236                     	| 14             	|
| FINAL_VESSEL_ONLY_hepaticvessel_352 	| nnunet+GATSegDiff  	| Success 	| 1632.46                	| 30                	| 1.33            	| 19.91                    	| 197                     	| 15             	|
| FINAL_VESSEL_ONLY_hepaticvessel_354 	| nnunet+GATSegDiff  	| Success 	| 897.42                 	| 17                	| 1.242           	| 20.87                    	| 129                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_360 	| nnunet+GATSegDiff  	| Success 	| 1669.65                	| 29                	| 1.246           	| 23.52                    	| 194                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_364 	| nnunet+GATSegDiff  	| Success 	| 1497.79                	| 25                	| 1.275           	| 23.4                     	| 176                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_357 	| nnunet+GATSegDiff  	| Success 	| 1627.41                	| 30                	| 1.288           	| 21.99                    	| 240                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_366 	| nnunet+GATSegDiff  	| Success 	| 1481.25                	| 31                	| 1.241           	| 17.03                    	| 186                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_370 	| nnunet+GATSegDiff  	| Success 	| 2029.24                	| 53                	| 1.278           	| 17.49                    	| 230                     	| 19             	|
| FINAL_VESSEL_ONLY_hepaticvessel_355 	| nnunet+GATSegDiff  	| Success 	| 2049.2                 	| 51                	| 1.23            	| 18.3                     	| 252                     	| 19             	|
| FINAL_VESSEL_ONLY_hepaticvessel_377 	| nnunet+GATSegDiff  	| Success 	| 1237.44                	| 23                	| 1.577           	| 18.2                     	| 184                     	| 7              	|
| FINAL_VESSEL_ONLY_hepaticvessel_381 	| nnunet+GATSegDiff  	| Success 	| 1328.91                	| 26                	| 1.227           	| 17.96                    	| 180                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_365 	| nnunet+GATSegDiff  	| Success 	| 1660.81                	| 36                	| 1.204           	| 20.01                    	| 215                     	| 15             	|
| FINAL_VESSEL_ONLY_hepaticvessel_376 	| nnunet+GATSegDiff  	| Success 	| 1709.36                	| 35                	| 1.18            	| 20.35                    	| 223                     	| 9              	|
| FINAL_VESSEL_ONLY_hepaticvessel_392 	| nnunet+GATSegDiff  	| Success 	| 2321.24                	| 42                	| 1.242           	| 20.36                    	| 230                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_380 	| nnunet+GATSegDiff  	| Success 	| 3169.91                	| 80                	| 1.186           	| 19.94                    	| 261                     	| 26             	|
| FINAL_VESSEL_ONLY_hepaticvessel_394 	| nnunet+GATSegDiff  	| Success 	| 2152.04                	| 39                	| 1.309           	| 23.14                    	| 209                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_387 	| nnunet+GATSegDiff  	| Success 	| 2869.57                	| 71                	| 1.202           	| 18.05                    	| 258                     	| 21             	|
| FINAL_VESSEL_ONLY_hepaticvessel_390 	| nnunet+GATSegDiff  	| Success 	| 3700.21                	| 88                	| 1.164           	| 19.27                    	| 254                     	| 22             	|
| FINAL_VESSEL_ONLY_hepaticvessel_403 	| nnunet+GATSegDiff  	| Success 	| 1757.76                	| 43                	| 1.256           	| 17.94                    	| 210                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_399 	| nnunet+GATSegDiff  	| Success 	| 1404.42                	| 29                	| 1.25            	| 19.78                    	| 204                     	| 6              	|
| FINAL_VESSEL_ONLY_hepaticvessel_410 	| nnunet+GATSegDiff  	| Success 	| 2552.08                	| 44                	| 1.248           	| 23.63                    	| 241                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_395 	| nnunet+GATSegDiff  	| Success 	| 2867.42                	| 63                	| 1.155           	| 20.63                    	| 240                     	| 24             	|
| FINAL_VESSEL_ONLY_hepaticvessel_414 	| nnunet+GATSegDiff  	| Success 	| 1807.34                	| 50                	| 1.199           	| 16.43                    	| 239                     	| 16             	|
| FINAL_VESSEL_ONLY_hepaticvessel_417 	| nnunet+GATSegDiff  	| Success 	| 1705.63                	| 40                	| 1.263           	| 19.83                    	| 211                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_418 	| nnunet+GATSegDiff  	| Success 	| 1539.3                 	| 34                	| 1.351           	| 17.3                     	| 183                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_421 	| nnunet+GATSegDiff  	| Success 	| 1600.48                	| 30                	| 1.23            	| 22.54                    	| 192                     	| 11             	|
| FINAL_VESSEL_ONLY_hepaticvessel_415 	| nnunet+GATSegDiff  	| Success 	| 2459.36                	| 58                	| 1.239           	| 19.37                    	| 257                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_435 	| nnunet+GATSegDiff  	| Success 	| 1181.13                	| 21                	| 1.327           	| 20.02                    	| 155                     	| 8              	|
| FINAL_VESSEL_ONLY_hepaticvessel_426 	| nnunet+GATSegDiff  	| Success 	| 3096.45                	| 81                	| 1.19            	| 18.32                    	| 244                     	| 21             	|
| FINAL_VESSEL_ONLY_hepaticvessel_436 	| nnunet+GATSegDiff  	| Success 	| 2066.22                	| 41                	| 1.327           	| 20.26                    	| 223                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_428 	| nnunet+GATSegDiff  	| Success 	| 1576.52                	| 31                	| 1.226           	| 20.74                    	| 206                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_430 	| nnunet+GATSegDiff  	| Success 	| 2767.91                	| 76                	| 1.181           	| 17.52                    	| 247                     	| 19             	|
| FINAL_VESSEL_ONLY_hepaticvessel_451 	| nnunet+GATSegDiff  	| Success 	| 1465.84                	| 28                	| 1.42            	| 22.21                    	| 178                     	| 5              	|
| FINAL_VESSEL_ONLY_hepaticvessel_452 	| nnunet+GATSegDiff  	| Success 	| 2768.83                	| 57                	| 1.266           	| 20.82                    	| 247                     	| 18             	|
| FINAL_VESSEL_ONLY_hepaticvessel_439 	| nnunet+GATSegDiff  	| Success 	| 2457.67                	| 59                	| 1.209           	| 19.2                     	| 248                     	| 21             	|
| FINAL_VESSEL_ONLY_hepaticvessel_457 	| nnunet+GATSegDiff  	| Success 	| 1690.43                	| 44                	| 1.241           	| 17.25                    	| 195                     	| 10             	|
| FINAL_VESSEL_ONLY_hepaticvessel_448 	| nnunet+GATSegDiff  	| Success 	| 2590.03                	| 52                	| 1.192           	| 20.56                    	| 249                     	| 17             	|
| FINAL_VESSEL_ONLY_hepaticvessel_450 	| nnunet+GATSegDiff  	| Success 	| 1872.24                	| 40                	| 1.182           	| 19.92                    	| 222                     	| 13             	|
| FINAL_VESSEL_ONLY_hepaticvessel_460 	| nnunet+GATSegDiff  	| Success 	| 2016.58                	| 42                	| 1.182           	| 21.01                    	| 224                     	| 16             	|

The results of the ensemble of the GATSegDiff and nnUNet predictions on the 3D-ircadb-01 Dataset. 

| case_id                   | model_architecture | status  | total_vessel_length_mm | bifurcation_count | mean_tortuosity | average_branch_length_mm | fragmentation_errors_h0 | false_loops_h1 |
|---------------------------|--------------------|---------|------------------------|-------------------|-----------------|--------------------------|-------------------------|----------------|
| ENSEMBLE_patient_004_mask | nnUNet+GATSegDiff  | Success | 2904.05                | 113               | 1.361           | 13.57                    | 245                     | 8              |
| ENSEMBLE_patient_001_mask | nnUNet+GATSegDiff  | Success | 1824.72                | 131               | 1.398           | 7.63                     | 230                     | 2              |
| ENSEMBLE_patient_002_mask | nnUNet+GATSegDiff  | Success | 1459.63                | 142               | 1.244           | 5.02                     | 206                     | 5              |
| ENSEMBLE_patient_005_mask | nnUNet+GATSegDiff  | Success | 4979.21                | 595               | 1.266           | 4.77                     | 260                     | 19             |
| ENSEMBLE_patient_003_mask | nnUNet+GATSegDiff  | Success | 2270.1                 | 195               | 1.285           | 6.47                     | 242                     | 11             |
| ENSEMBLE_patient_006_mask | nnUNet+GATSegDiff  | Success | 2552.54                | 201               | 1.388           | 6.12                     | 238                     | 9              |
| ENSEMBLE_patient_008_mask | nnUNet+GATSegDiff  | Success | 1665.89                | 181               | 1.401           | 5.43                     | 225                     | 6              |
| ENSEMBLE_patient_009_mask | nnUNet+GATSegDiff  | Success | 4834                   | 306               | 1.34            | 8.5                      | 260                     | 14             |
| ENSEMBLE_patient_010_mask | nnUNet+GATSegDiff  | Success | 1884.52                | 123               | 1.25            | 7.79                     | 215                     | 6              |
| ENSEMBLE_patient_007_mask | nnUNet+GATSegDiff  | Success | 2586.06                | 176               | 1.357           | 7.56                     | 237                     | 6              |
| ENSEMBLE_patient_011_mask | nnUNet+GATSegDiff  | Success | 2458.23                | 197               | 1.33            | 6.75                     | 244                     | 8              |
| ENSEMBLE_patient_014_mask | nnUNet+GATSegDiff  | Success | 1493.33                | 97                | 1.241           | 8.07                     | 208                     | 4              |
| ENSEMBLE_patient_013_mask | nnUNet+GATSegDiff  | Success | 2726.68                | 187               | 1.352           | 7.6                      | 248                     | 7              |
| ENSEMBLE_patient_015_mask | nnUNet+GATSegDiff  | Success | 2024.11                | 162               | 1.384           | 6.02                     | 220                     | 6              |
| ENSEMBLE_patient_017_mask | nnUNet+GATSegDiff  | Success | 1605.93                | 133               | 1.204           | 6.18                     | 192                     | 5              |
| ENSEMBLE_patient_018_mask | nnUNet+GATSegDiff  | Success | 2383.07                | 91                | 1.446           | 13.31                    | 228                     | 5              |
| ENSEMBLE_patient_016_mask | nnUNet+GATSegDiff  | Success | 1738.37                | 129               | 1.401           | 7.24                     | 208                     | 3              |
| ENSEMBLE_patient_019_mask | nnUNet+GATSegDiff  | Success | 6735.55                | 444               | 1.348           | 7.68                     | 273                     | 17             |
| ENSEMBLE_patient_012_mask | nnUNet+GATSegDiff  | Success | 2132.04                | 216               | 1.332           | 5.66                     | 241                     | 8              |
| ENSEMBLE_patient_020_mask | nnUNet+GATSegDiff  | Success | 9314.63                | 730               | 1.37            | 7.07                     | 316                     | 20             |

To then understandand what these results mean for these models we've tested, we set up biologivan invarant benchmarks using Murray's Law, Murray's deviatation, and area-expansion. Using our TDA results, we predictied how the statistics affected the numbers seen in each biophysical metric, and we compared it to the baseline metrics we set universally. The three benchmarks we used were "Valid Anatomy", which signifies healthy blood vessels that fit the ideal range for each invariant; "Plausible Tumor" which is mean to account for the reasonable range in which a model's prediction which move away from the ideal scenario given that most of the cases are dealing with hypervascular tumors that significantly disotrt natural physics properties; and lasty, we use "Impossible Hallucination" to represent those prediction's that would move past the plausibily range and generally signifies that a model likely made a serious error. 

The results of the nnUNet predictions on the MSD Tak08 Hepatic Vessel Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/nnUNet-on-MSD_Audited_Clinical_Report.csv)

The results of the GATSegDiff on the MSD Tak08 Hepatic Vessel Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/GATSegDiff-on-MSD_Audited_Clinical_Report.csv)

The results of the nUNet predictions on the original 3D-ircadb-01 Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/nnUNet-on-3Dircadb_Audited_Clinical_Report.csv)

The results of the GATSegDiff predictions on the original 3D-ircadb-01 Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/GATSegDiff-on-3Dircadb_Audited_Clinical_Report.csv)

The results of the ensemble of the GATSegDiff and nnUNet predictions on the MSD Task08 HepaticVessel Dataset.

![alt text](https://github.com/Ivy300-byte/VesselSemble/blob/main/GATSegDff-and-nnUNet-on-MSD_Audited_Clinical_Report.csv)

The results of the ensemble of the GATSegDiff and nnUNet predictions on the 3D-ircadb-01 Dataset.

| Case_ID     	| H1_False_Loops 	| Frag_Errors_H0 	| Predicted_Murray_Score 	| Predicted_Area_Expansion 	| Physics_Shift_Score 	| Clinical_Audit_Status    	| Anatomical_Classification      	| Surgical_Directive                                                            	|
|-------------	|----------------	|----------------	|------------------------	|--------------------------	|---------------------	|--------------------------	|--------------------------------	|-------------------------------------------------------------------------------	|
| PATIENT_004 	| 8              	| 245            	| 0.769                  	| 1.297                    	| 0.231               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_001 	| 2              	| 230            	| 0.519                  	| 1.269                    	| 0.481               	| IMPOSSIBLE HALLUCINATION 	| Biophysical Invariant Collapse 	| CRITICAL REJECT: Structure breaks physical laws; software model error.        	|
| PATIENT_002 	| 5              	| 206            	| 0.488                  	| 1.268                    	| 0.512               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_005 	| 19             	| 260            	| 0.839                  	| 1.269                    	| 0.161               	| IMPOSSIBLE HALLUCINATION 	| Biophysical Invariant Collapse 	| CRITICAL REJECT: Structure breaks physical laws; software model error.        	|
| PATIENT_003 	| 11             	| 242            	| 0.658                  	| 1.279                    	| 0.342               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_006 	| 9              	| 238            	| 0.694                  	| 1.287                    	| 0.306               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_008 	| 6              	| 225            	| 0.509                  	| 1.281                    	| 0.491               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_009 	| 14             	| 260            	| 0.853                  	| 1.282                    	| 0.147               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_010 	| 6              	| 215            	| 0.617                  	| 1.272                    	| 0.383               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_007 	| 6              	| 237            	| 0.685                  	| 1.278                    	| 0.315               	| IMPOSSIBLE HALLUCINATION 	| Biophysical Invariant Collapse 	| CRITICAL REJECT: Structure breaks physical laws; software model error.        	|
| PATIENT_011 	| 8              	| 244            	| 0.664                  	| 1.278                    	| 0.336               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_014 	| 4              	| 208            	| 0.505                  	| 1.269                    	| 0.495               	| IMPOSSIBLE HALLUCINATION 	| Biophysical Invariant Collapse 	| CRITICAL REJECT: Structure breaks physical laws; software model error.        	|
| PATIENT_013 	| 7              	| 248            	| 0.692                  	| 1.279                    	| 0.308               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_015 	| 6              	| 220            	| 0.621                  	| 1.282                    	| 0.379               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_017 	| 5              	| 192            	| 0.578                  	| 1.265                    	| 0.422               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_018 	| 5              	| 228            	| 0.7                    	| 1.301                    	| 0.3                 	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_016 	| 3              	| 208            	| 0.556                  	| 1.275                    	| 0.444               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_019 	| 17             	| 273            	| 0.895                  	| 1.279                    	| 0.105               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_012 	| 8              	| 241            	| 0.603                  	| 1.277                    	| 0.397               	| PLAUSIBLE PATHOLOGY      	| Tumor Neoangiogenesis Bed      	| APPROVE: Complex tumor vascularization captured; preserve for clinical trace. 	|
| PATIENT_020 	| 20             	| 316            	| 0.905                  	| 1.275                    	| 0.095               	| IMPOSSIBLE HALLUCINATION 	| Biophysical Invariant Collapse 	| CRITICAL REJECT: Structure breaks physical laws; software model error.        	|



