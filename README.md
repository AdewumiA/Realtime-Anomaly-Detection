# Anomaly-Detection

Anomaly detection in Time series data.  

## **Method 1: Rolling Window Rapid distance-based via sampling**

This method is based on the research paper by [Sugiyama & Borgwardt (2013)](https://papers.nips.cc/paper_files/paper/2013/file/d296c101daa88a51f6ca8cfc1ac79b50-Paper.pdf). 

It was slightly modified to use rolling window for the timeseries data. Choice of window size was based on the research paper. This method has the following advantages:

* **Scalable**: the time complexity is linear in the number of data points,
* **Effective**: it is empirically shown to be the most effective on average among existing distance-based outlier detection methods,
* **Easy to use**: it requires only one parameter, the number of samples, and a small sample size (the default value is 60), and
* **Rolling window**: It ensures anomaly detection is done on the recently available data in the time series. 

## **Method 2: Isolation Forest**

The feature importance of the isolation forest was determined based on the methodology presented in ["Explainable machine learning in industry 4.0: evaluating feature importance in anomaly detection to enable root cause analysis."](https://ieeexplore.ieee.org/abstract/document/8913901) by Carletti et al. (2019). 
The unofficial implementation of the code can be found on [https://github.com/AdewumiA/diffi_df](https://github.com/AdewumiA/diffi_df) which was forked from [https://github.com/britojr/diffi](https://github.com/britojr/diffi). 

## **Method 3: Attention Free Autoencoder**

This method is based on the research work [An Attention Free Conditional Autoencoder For Anomaly Detection in Cryptocurrencies](https://arxiv.org/abs/2304.10614) by Inzirillo and De Villelongue (2023).
