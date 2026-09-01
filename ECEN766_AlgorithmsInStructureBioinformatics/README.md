# Cross Modality Learning on Proteins.
## Abstract
**Motivation:** Discovering sequence-structure-function relationship on protein molecules remains a great challenge for bioinformatics scientists. While the protein functions are highly related to the structures, the structures are also folded by protein sequences of 20 kinds of amino acids following some complex and intractable physical principles. With the quickly accumulating data on proteins sequences and structures, more and more people began to focus on data-driven methods and apply artificial intelligence models to learn the relationship automatically. On the other hand, to represent proteins there are also various modalities that reflect the information on different aspects, so a question this project is addressing is: will the information contained in other modalities help to learn the sequence-structure relationship?  
**Results:** This project focused on the fold prediction problem which is to learn the structure category a protein would belong to. The fold prediction model was based on the state-of-the-art model DeepSF but with less input features. At the same time, another modality of the proteins, the secondary structure (SS) is also considered as its related prediction model was constructed and combined to the fold prediction model in order to get a better performance. The Top-1 accuracy of the DeepSF with only protein sequences as the input was 0.122, while it reached 0.502 when the SS was also taken as the input. The best SS prediction model attained an accuracy of 0.833, and when it was combined to the DeepSF model the Top-1 accuracy was improved to 0.131.

## Environment
To install the environment for this project, install the [Anaconda](https://www.anaconda.com/) and go the the ***Environment/*** directory to run the following script:
```
conda env create -f CrossModality.yml
```

## Data
In order to get the processed data for the fold prediction model, please follow the steps below:
* Go to the ***Data/Scripts/*** directory.
* Run the following scripts:
'''
mkdir ../Datasets/For_Seq_Stru/features
mkdir ../Datasets/For_Seq_Stru/features_ss
mkdir ../Datasets/For_Seq_Stru/features_pre
'''
* Open ***Data_Process.ipynb*** and follow the steps after ***"Datasets for Seq-Fold models:"***.
* To get the mixed data for DeepSF (AA + SS) with different portions of the real SS, run:
```
mkdir ../Datasets/For_Seq_Stru/features_mix_<portion>
python SS_mix.py <portion>
```
portion is an int number between 0 and 100.

## Train the Fold Prediction Model (DeepSF)
* Go to the ***Seq_Fold/*** directory.
* Load the environment:
```
source activate CrossModality 
```
* To train the DeepSF (AA):
```
python DeepSF.py
```
* To train the DeepSF (AA + SS):
```
python DeepSF_ss.py
```
* To train the DeepSF (AA + SS<sub>pre</sub>):
```
python DeepSF_ssPre.py
```
* To train the DeepSF (AA + SS) with different portions of the real SS:
```
python DeepSF_mix.py  <portion>
```
