# DOMI


###  Detecting Outlier Machine Instances through One Dimensional CNN Gaussian Mixture Variational AutoEncoder

DOMI is a VAE-based model which glues one Dimensional Convolution Neural Network and Gaussian Mixture Variational auto-encoder. 
It aims at detecting outlier machine instances and its core idea is to learn the normal patterns of multivariate time series
and use the reconstruction probability to do outlier judgment. 
Moreover, for a detected outlier machine instance, DOMI provides interpretation based on reconstruction probability changes of univaraite time series.



## Getting Started

#### Clone the repo

```
git clone https://github.com/Tsinghuasuya/DOMI_code
```

#### Get data from github and unzip 

```
git lfs clone https://github.com/Tsinghuasuya/DOMI_dataset && cd DOMI_dataset && unzip publicDataset.zip  && cd  ../DOMI_code
```


#### Install dependencies (with python 3.6) 

(virtualenv is recommended)

```shell
pip install -r requirements.txt
```


#### Run the code

```
python domi.py
```

If you want to change the default configuration, you can edit `ExpConfig` in `config.py` or 
overwrite the config in `domi.py` using command line args. For example:

```
python domi.py --noExp=2 --max_epoch=100 --initial_lr=0.0001 
```


## Result

After running the programmings, you can get the output in the file directory that you set in the config. For each instance, you can get the total score and score of each univariate time series. 
All the results are in the folder `{config.result_dir}/`, with trained model in `{config.result_dir}/DOMI_{noExp}.model`, the output and config of DOMI in the folder `{config.result_dir}/DOMI_{noExp}/`, 
and the detailed detection results are in the folder `DOMI_{noExp}/result_summay/`. It's made up by the following parts:
* `OutlierScores_metric.txt`: score of each univariate time series for instance in the testing dataset.
* `OutlierScores.txt`: score for each instance in the testing dataset.
* `MetricResult.txt`: interpretation using the univariate time series of machine instances.
* `PRF.txt`: summary of the overall statistics, including expected score of each metric and F1-score, recall, precision. 
