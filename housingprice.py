import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(r'C:\Users\SIDDHANT\Desktop\ML\train.csv')
X_train = train.iloc[:, :-1].values
Y_train = train.iloc[:, 80].values
test=pd.read_csv(r'C:\Users\SIDDHANT\Desktop\ML\test.csv')
X_train = test.iloc[:, :-1]


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
kl_train=train.iloc[:,:-1]
train.Alley.describe()
print("unique values are:",train.Alley.unique())
print(train.Alley.value_counts(),"\n")
train['enc_Alley']=pd.get_dummies(train.Alley,drop_first=True)
test['enc_Alley']=pd.get_dummies(test.Alley,drop_first=True)
print(train.enc_Alley.value_counts())

print("unique values are:",train.LotFrontage.unique())
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()
train['enc_street']=pd.get_dummies(train.Street,drop_first=True)
test['enc_street']=pd.get_dummies(test.Street,drop_first=True)
print(train.enc_street.value_counts())

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

sum(data.isnull().sum()!=0)

h=[2,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,27,28,29,39,40,41,52,54,56,59,62,71,77,78,79]




#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer = imputer.fit(train[:,6:7])
train[:,6:7]=imputer.transform(train[:,6:7])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_train = LabelEncoder()

for i in h:
    train[:,i]=labelencoder_train.fit_transform(train[:,i])  
    onehotencoder = OneHotEncoder(categorical_features = [i])

#X = onehotencoder.fit_transform(X).toarray()
#onehotencoder = OneHotEncoder(categorical_features = [2])
