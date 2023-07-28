### importing libraries
import pandas as pd 
import numpy as np
#importing dataset
df=pd.read_csv('D:/data science/assignments_csv/ToyotaCorolla.csv')
df.columns
df.drop(['Id', 'Model', 'Mfg_Month', 'Mfg_Year',
       'Fuel_Type',  'Met_Color', 'Color', 'Automatic',
       'Cylinders',   'Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'],axis=1,inplace=True)

df
df.columns
df.info()
df.describe()
df.isna().sum()
#### correlation analysis
df.corr()
##############################################################################
###### EDA
### scatterplot b/w price and Age
import matplotlib.pyplot as plt
plt.scatter(x=df[['Age_08_04']],y=df['Price'],color='red')
plt.xlabel('Age_08_04')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and KM
import matplotlib.pyplot as plt
plt.scatter(x=df[['KM']],y=df['Price'],color='red')
plt.xlabel('KM')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and HP
import matplotlib.pyplot as plt
plt.scatter(x=df[['HP']],y=df['Price'],color='red')
plt.xlabel('HP')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and cc
import matplotlib.pyplot as plt
plt.scatter(x=df[['cc']],y=df['Price'],color='red')
plt.xlabel('cc')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and Doors
import matplotlib.pyplot as plt
plt.scatter(x=df[['Doors']],y=df['Price'],color='red')
plt.xlabel('Doors')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and Gears
import matplotlib.pyplot as plt
plt.scatter(x=df[['Gears']],y=df['Price'],color='red')
plt.xlabel('Gears')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and tax
import matplotlib.pyplot as plt
plt.scatter(x=df[['Quarterly_Tax']],y=df['Price'],color='red')
plt.xlabel('Quarterly_Tax')
plt.ylabel('Price')
plt.show()
### scatterplot b/w price and weight
import matplotlib.pyplot as plt
plt.scatter(x=df[['Weight']],y=df['Price'],color='red')
plt.xlabel('Weight')
plt.ylabel('Price')
plt.show()

############ boxplot for all x variables
df.boxplot(column='Age_08_04',vert=True)
df.boxplot(column='KM',vert=True)
df.boxplot(column='HP',vert=True)
df.boxplot(column='cc',vert=True)
df.boxplot(column='Doors',vert=True)
df.boxplot(column='Gears',vert=True)
df.boxplot(column='Quarterly_Tax',vert=True)
df.boxplot(column='Weight',vert=True)

########### histogram for all y variables
df['Weight'].hist()
df['Age_08_04'].hist()
df['KM'].hist()
df['HP'].hist()
df['cc'].hist()
df['Doors'].hist()
df['Gears'].hist()
df['Quarterly_Tax'].hist()

#####################################################################
##### Multicoloniarity 
import statsmodels.formula.api as smf
Model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
Model.summary()
#### variance influence factror
import statsmodels.formula.api as smf
Model=smf.ols('HP~KM+Age_08_04+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
R2=Model.rsquared
VIF=1/1-R2
print('variance influencing factor',VIF.round(3))

######## Resididual analysis
Model.resid
Model.resid.hist()

#### Test for normality of residuals 
import matplotlib.pyplot as plot 
import statsmodels.api as sm
qqplot=sm.qqplot(Model.resid,line='q')
plt.title('Q-Q plot of residuals')
plt.show()

Model.fittedvalues ### predicted values 
Model.resid ### error values
### Pattern checking
import matplotlib.pyplot as plt
plt.scatter(Model.fittedvalues,Model.resid)
plt.title('Residual plot')
plt.xlabel('Model fittedvalues')
plt.ylabel('Model residual')
plt.show()

# Model Deletion Diagnostics
## Detecting Influencers/Outliers

### cooks distance
Model_influence = Model.get_influence()
(cooks, pvalue) = Model_influence.cooks_distance

cooks = pd.DataFrame(cooks)

#Plot the influencers values using stem plot
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0],5))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

# index and value of influencer where c is more than .5
cooks[0][cooks[0]>0.5]

## High Influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(Model)
plt.show()
### Leverage Cutoff
k = df.shape[1]
n = df.shape[0]
leverage_cutoff = (3*(k + 1)/n)
leverage_cutoff
cooks[0][cooks[0]>leverage_cutoff]
df.shape
### droping the rows which are under levarage cutoff
df.drop([8,10,11,12,13,14,15,16,49,53,80,141,221,601,654,956,960,991,1044],inplace=True)
df.shape

######################################################################################
### splitting variables
Y=df['Price']
X=df.iloc[:,1:]
X
##### data Transformations 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X)
ss_x=pd.DataFrame(ss_x)
ss_x.columns=['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']
ss_x

##### data partation
###### Testing and Training 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.70)
X_train.shape
X_test.shape
Y_train.shape
X_test.shape

### Selecting few models
### model fitting for Linear regression
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)

### model predictions
Y_train_pred=LR.predict(X_train)
Y_test_pred=LR.predict(X_test)

### matrics
from sklearn.metrics import mean_squared_error
mse_train=np.sqrt(mean_squared_error(Y_train_pred,Y_train))
mse_test=np.sqrt(mean_squared_error(Y_test_pred,Y_test))
print('Training mean sqared error',mse_train.round(2))
print('Test mean sqared error',mse_test.round(2))


#### Cross validation for all chosen models
### K-Fold validation
from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)

Training_mse.append(np.sqrt(mean_squared_error(Y_train,Y_train_pred)))
Test_mse.append(np.sqrt(mean_squared_error(Y_test,Y_test_pred)))

import numpy as np
print('training mean squared error:',np.mean(Training_mse).round(3))    
print('test mean squared error:',np.mean(Test_mse).round(3))
 
## Shrinking Methods
### Ridge Regression
from sklearn.linear_model import Lasso
LS=Lasso(alpha=8)

LS.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(LR.coef_)
#a1=pd.DataFrame(LS.coef_)
#a3=pd.DataFrame(LS.coef_)
#a5=pd.DataFrame(LS.coef_)
a8=pd.DataFrame(LS.coef_)
df_lasso=pd.concat([d1,d2,a1,a3,a5,a8],axis=1)
df_lasso.columns=['names','LR','alpha1','alpha','alpha5','alpha8']
df_lasso

#################################################################
#### droping 5th column
df
X=df.iloc[:,1:]
X
X_new=X.drop(X.columns[[5]],axis=1)
X_new


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X_new)

from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(ss_x):
    X_train,X_test=ss_x[train_index],ss_x[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)

Training_mse.append(np.sqrt(mean_squared_error(Y_train,Y_train_pred)))
Test_mse.append(np.sqrt(mean_squared_error(Y_test,Y_test_pred)))

import numpy as np
print('training mean squared error:',np.mean(Training_mse).round(3))    
print('test mean squared error:',np.mean(Test_mse).round(3)) 
#######################################################################################
############### Final Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
ss_x    = ss.fit_transform(X_new)
LR.fit(X_new,Y)
LR.intercept_
LR.coef_
###### Final Model Fitted Values
dt=pd.DataFrame({'Age_08_04':25,'KM':50000,'HP':100,'cc':2500,'Doors':4,'Quarterly_Tax':250,'Weight':1200},index=[0])
t1=LR.predict(dt)
t1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            q`1111111111111111111111111111111111111111111111111111111111q






.




















































































































































































































