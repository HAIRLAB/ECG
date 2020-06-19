# Python Scripts
This is the python scripts for data preprocessing and modelling of the following paper in *The lancet Digital Health*:

Zhu, Hongling, et al. "Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning: a cohort study." The Lancet Digital Health (2020). [Open access.](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(20)30107-2/fulltext) 

Bibtex citation:

	@article{zhu2020automatic,
	  title={Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning: a cohort study},	
	  author={Zhu, Hongling and Cheng, Cheng and Yin, Hang and Li, Xingyi and Zuo, Ping and Ding, Jia and Lin, Fan and Wang, Jingyi and Zhou, Beitong and Li, Yonge and others},
	  journal={The Lancet Digital Health},
	  year={2020},
	  publisher={Elsevier}
	}


## Files

* `model_training.py `
	* Script for training the diagnosis network
* `ecg_preprocessing.py`
	* Script for the pre-processing procedure of the ECG recordings
* `modelbuild.py`
	* Network structrue for the multi-label diagnosis model
* `config.json`
	* root directory and hyper parameters


## Test dataset
The test dataset from TongjiHospital of this study is publicly available at [Mendeley Data](https://data.mendeley.com/datasets/6jd4rn2z9x/1).

The Independent China Physiological SignalChallenge dataset is a public dataset available at: [http://2018.icbeb.org/Challenge.html.](http://2018.icbeb.org/Challenge.html.)



