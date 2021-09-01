# FailurePrediction-NasaTurbofan
* Predicted the remaining useful life of turbofan based on sensor information using regression and deep learning models.
* Performed EDA on FD001 dataset. Removed unnecessary sensors and setting conditions.
* Fitted base model of linear regression. Also fitted random forest, xg_boost and time series models.
* Fitted deep learning models- MLP, LSTM.
* Calculated feature importance using SHAP values.

## Code and Resources Used
* **Python Version:** 3.8.5
* **Packages:** pandas, numpy, sklearn, xgboost, matplotlib, seaborn, tensorflow, statsmodels
* **GitHub Repo Ref:** https://github.com/kpeters/exploring-nasas-turbofan-dataset
* **Kaggle Ref1:** https://www.kaggle.com/vinayak123tyagi/damage-propagation-modeling-for-aircraft-engine 
* **Kaggle Ref2:** https://www.kaggle.com/sanchitapaul/nasa-turbofan-degradation-model

## EDA
Plotted frequency of RUL, Sensors against RUl and Correlation between sensors:
![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/RUL.PNG)
![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/sensor14.PNG)
![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/heatmap.PNG)

## Model Building
* Split the dataset into 70:30 for train and test respectively based on unique ID so that data from same engine doesn't divide into train and test.
* Used RMSE and R2 score for evaluation.
* Used six models:
  * Multilinear regression - Baseline for the model.
  * Random Forest -Hypertuned the parameters using RandomizedSearchCV.
  * XGBoost -Hypertuned the parameters using RandomizedSearchCV.
  * TimeSeries model- Converted data to staionary and added lags.
  * MLP- Used deep learning to check prediction.
  * LSTM model - good with sequential data.

### Model performance
LSTM performed the best:

![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/modelperformance.PNG)
![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/comparision.PNG 'LSTM')

### Feature Importance

![alt text](https://github.com/Ajay-rai/FailurePrediction-NasaTurbofan/blob/main/img/shap.PNG)

## Conclusion and Future Recommendation
* LSTM model predicts the RUL quite well with 11.3% improvement to linear regression(base model). Increasing epochs can improve it further.
* Sensor 14 and 11 impacts the RUL the most.
* Threshold value of RUL can be set and problem can be solved using a classification model as well.
* LSTM model will work for other data sets as well with more complicated settings.
* Better feature selection can improve the model further. However, domain knowledge will play a important role.
