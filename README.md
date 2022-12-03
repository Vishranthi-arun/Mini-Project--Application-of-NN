# Mini-Project--Application-of-NN

## Project Title:
Stock market prediction

## Project Description 
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%.
Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.

## Algorithm:
1. import the necessary pakages.
2. install the csv file
3. using the for loop and predict the output
4. plot the graph 
5. analyze the regression bar plot

## Google Colab Link:
https://colab.research.google.com/drive/1rknMNlbLphgS6ObhFSfGUWMfB-S_kPIE?usp=sharing

## Program:
import the necessary pakages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```
install the csv file
```
df = pd.read_csv('/content/Tesla.csv')
df.head()
```
```
df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()

```
## Output:
![image](https://user-images.githubusercontent.com/93427278/205438173-9196c1d9-d1fa-4e74-b266-d23fee3401bd.png)




![image](https://user-images.githubusercontent.com/93427278/205438196-596a76e9-86c8-47ea-8187-c13564263d45.png)





![image](https://user-images.githubusercontent.com/93427278/205438259-95f88ec2-b61e-4716-b1da-8b05c0d3b0e3.png)





![image](https://user-images.githubusercontent.com/93427278/205438320-23d2e985-9b69-4df8-b772-75fbe0922d7c.png)




![image](https://user-images.githubusercontent.com/93427278/205438359-9d418e17-b3dc-49ed-bc45-eb1c819d6006.png)





![image](https://user-images.githubusercontent.com/93427278/205438384-d878e425-d958-4533-9cb7-c993fb6604ae.png)






![image](https://user-images.githubusercontent.com/93427278/205438400-6ad9ce27-78a6-413c-a283-965f5d39890b.png)






![image](https://user-images.githubusercontent.com/93427278/205438546-e1d94bc0-c469-4934-bb74-dac982f57320.png)





![image](https://user-images.githubusercontent.com/93427278/205438568-19925be5-6105-4027-9474-f1d3942f4aaa.png)






![image](https://user-images.githubusercontent.com/93427278/205438602-710361e2-22ab-4c92-a7c2-1b107e6820ed.png)






![image](https://user-images.githubusercontent.com/93427278/205438690-1e1ae744-67a8-40c0-a1e4-6b897732cbbc.png)






![image](https://user-images.githubusercontent.com/93427278/205438720-4a2f6508-2079-43dd-be33-5190caeccf53.png)





![image](https://user-images.githubusercontent.com/93427278/205438746-790653fa-ec49-4be9-bb90-fccba9a60cc0.png)






![image](https://user-images.githubusercontent.com/93427278/205438768-4365e122-bef2-42d8-9afc-73ee968b9d94.png)






![image](https://user-images.githubusercontent.com/93427278/205438796-b2e22e65-0ebe-4a05-8524-4b38f86c26e9.png)



## Advantage :
Python is the most popular programming language in finance. 
Because it is an object-oriented and open-source language, it is used by many large corporations,
including Google, for a variety of projects. Python can be used to import financial data such as
stock quotes using the Pandas framework.

## Result:
Thus, stock market prediction is implemented successfully.
