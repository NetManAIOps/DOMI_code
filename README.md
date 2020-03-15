# DOMI


###  Detecting Outlier Machine Instances through One Dimensional CNN Gaussian Mixture Variational AutoEncoder

DOMI is a VAE-based model which glues one Dimensional Convolution Neural Network and Gaussian Mixture Variational auto-encoder. It aims at detecting outlier machine instances and its core idea is to learn the normal patterns of multivariate time series and use the reconstruction probability to do outlier judgment. 
Moreover, for a detected outlier machine instance, DOMI provides interpretation based on reconstruction probability changes.



## Getting Started

#### Clone the repo

```
git clone https://github.com/ijcai20/DOMI && cd DOMI
```

#### Get data

OMI_dataset (Outlier Machine Instances Dataset) is in folder `OMI_dataset`. 

#### Install dependencies (with python 3.5, 3.6) 

(virtualenv is recommended)

```shell
pip install -r requirements.txt
```


#### Run the code

```
python domi.py
```

If you want to change the default configuration, you can edit `ExpConfig` in `config.py` or overwrite the config in `domi.py` using command line args. For example:

```
python domi.py --noExp=2 --max_epoch=100 --initial_lr=0.0001 
```



## Data

### Dataset Information

| Dataset name| time points of each instance | univariate time series of each instance |  Instance matrix |
|:------:|:----:|:--------:|:-----:|
| OMI_dataset | 288 | 19 | 19 * 288 |
| **Training set size** |**<br/>Outlier Ratio in </br> Training set (%)** |**Testing set size**|**<br/>Outlier Ratio in </br> Testing set (%)**| 
|  54630 | 18.62 | 27315 | 22.26 |


### OMI_dataset

OMI_dataset (Outlier Machine Instances Dataset) is a server machine dataset collected from a top global Internet company. This dataset contains 1821 machines last for one and a half months, with 5-minute equal-spaced timestamps. Every instance named M-X@D-Y (means machine X at day Y) is a T * M matrix, where M and
T are the number of univariate time series and time points in one day, respectively. In our dataset, each machine is consituted of 19 metrics (i.e., M=19), and each day has 288 time points (i.e., T=288).

We divide the overall dataset into two parts, the first month for training and the second half month for testing. For the testing dataset, we provide labels for outlier machine instances, and interpretation labels for outlier instances.

Thus OMI_dataset is made up by the following parts:

* `train_data/`: Training set.
* `test_data/`: Testing set.
* `test_label/`: The labels of the testing set, which indicate whether an instance is an outlier. 
* `interpretation_label.txt`: The ground truth lists of univariate time series that contribute to outlier judgement.


## Result

After running the programmings, you can get the output in the file directory that you set in the config. For each instance, you can get the total score and score of each univariate time series. 
All the results are in the folder `{config.result_dir}/`, with trained model in `{config.result_dir}/DOMI_{noExp}.model`, the output and config of DOMI in the folder `{config.result_dir}/DOMI_{noExp}/`, 
and the detailed detection results are in the folder `DOMI_{noExp}/result_summay/`. It's made up by the following parts:
* `OutlierScores_metric.txt`: score of each univariate time series for instance in the testing dataset.
* `OutlierScores.txt`: score for each instance in the testing dataset.
* `MetricResult.txt`: interpretation using the univariate time series of machine instances.
* `PRF.txt`: summary of the overall statistics, including expected score of each metric and F1-score, recall, precision. 