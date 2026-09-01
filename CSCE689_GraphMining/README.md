# TAMU_CSCE689_CourseProject
Course project of CSCE689 (Graph Mining) in TAMU.

The data can be found in /Data/data.zip, and to run the code it need to be unzipped at first.

## GAE / GVAE
Firstly build the conda environment with the file *GVAE.yml* in the /Emviroment folder.\
Go to the folder /gae, and run the following command for training and validation, while *version* can be "gae" or "gvae":

```
python train_10FoldCV_gvae.py --m version
```

## NeoDTI
Go to the folder /NeoDTI, and run the following command for training and validation:
```
python NeoDTI_cv.py
```

## Graph Factorization
Go to the folder /gf, and run the command below for training and validation
```
python linkpre.py
```


