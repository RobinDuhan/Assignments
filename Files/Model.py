
#Importing the libraries
import pandas as pd 
import numpy as np

#Import this file
#Two functions for two models



def model_compressor(path):
    print("Model creation started")
    data = pd.read_csv(path)
    data.isnull().sum()
    data.dropna(inplace = True)
    data.drop(["Unnamed: 0"],axis = 1, inplace = True)
    NetF = list(data.columns)
    dat = pd.read_csv(path)
    #Two temperary lists/arrays for our use
    temp1=[]
    temp2=[]

    #Fiding Features whose skewness can be decreases
    skewFeature=[]

    for i in NetF:
        temp1.append(dat[i].skew())
        if (dat[i].min()) > 0 :
            dat[i] = dat[i].map(lambda s: np.log(s)if s > 0 else 0)
        #To check     
        if (dat[i].min()) <= 0 :
            dat[i] = dat[i].map(lambda s: -np.power(-s, 1./3) if s < 0 else np.power(s, 1./3)) 
            
        temp2.append(dat[i].skew())
    for n in range(4):
        if abs(temp1[n])>abs(temp2[n]):
            skewFeature.append(NetF[n])


    # In[9]:


    for i in skewFeature:
        if (data[i].min()) > 0 :
            data[i] = data[i].map(lambda k: np.log(k)if k > 0 else 0)
        #Checking
        if (data[i].min()) <= 0 :
            data[i] = data[i].map(lambda x: -np.power(-x, 1./3) if x < 0 else np.power(x, 1./3)) 


    # In[10]:


    #Outliers and removing them

    Ps = data.quantile(0.25)
    Ps1 = data.quantile(0.75)
    result = Ps1 - Ps

    data = data[~((data < (Ps - 2 * result)) |(data > (Ps1 + 2 * result))).any(axis=1)]
    data.shape
    print("Outliers and skewness removed")




    #Scalling the dataset

    from sklearn.preprocessing import StandardScaler,scale
    #we create an object of the class StandardScaler
    sc = StandardScaler() 

    col_to_scale = NetF
    data[col_to_scale] = sc.fit_transform(data[col_to_scale])
    data.head()
    print("Scaled the dataset")



    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, train_size = 0.8, test_size = 0.2, random_state = 0 )


    # First model for <b>GT Compressor decay state coefficient.</b>

    # In[29]:


    y_train = data_train.loc[:,data_train.columns == 'GT Compressor decay state coefficient.']
    X_train = data_train.loc[:, data_train.columns != 'GT Compressor decay state coefficient.']


    # In[30]:


    y_test = data_test.loc[:,data_test.columns == 'GT Compressor decay state coefficient.']
    X_test = data_test.loc[:, data_test.columns != 'GT Compressor decay state coefficient.']




    # <h4> Random Forest regressor </h4> <br>
    # 10 estimators 

    # In[35]:
    print("Random Forest Regression Model training")
    from sklearn.ensemble import RandomForestRegressor
    #Random forest 
    reg_rf = RandomForestRegressor(n_estimators=10, random_state=0)
    y_train_new = y_train["GT Compressor decay state coefficient."]
    reg_rf.fit(X_train, y_train_new)
    y_pred = reg_rf.predict(X_test)

    from sklearn.metrics import r2_score 
    print("R2 score is: ",r2_score(y_test, y_pred))

    #Increasing the estimators doesn't increase the r2 score that much so 10 estimators are optimal


    # In[36]:


    from sklearn import metrics

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    # In[37]:


    resid = y_test['GT Compressor decay state coefficient.'] - y_pred
    # Durbin watson test to check for autocorrelation in error terms, as its close to 2
    # That means our error terms are independent of each other

    from statsmodels.stats.stattools import durbin_watson
    print("Durbin watson autocorrelation cofficient:",durbin_watson(resid))
    return reg_rf




def model_turbo(path):
    print("Model creation started")
    data = pd.read_csv(path)
    data.isnull().sum()
    data.dropna(inplace = True)
    data.drop(["Unnamed: 0"],axis = 1, inplace = True)
    NetF = list(data.columns)
    dat = pd.read_csv(path)
    #Two temperary lists/arrays for our use
    temp1=[]
    temp2=[]

    #Fiding Features whose skewness can be decreases
    skewFeature=[]

    for i in NetF:
        temp1.append(dat[i].skew())
        if (dat[i].min()) > 0 :
            dat[i] = dat[i].map(lambda s: np.log(s)if s > 0 else 0)
        #To check     
        if (dat[i].min()) <= 0 :
            dat[i] = dat[i].map(lambda s: -np.power(-s, 1./3) if s < 0 else np.power(s, 1./3)) 
            
        temp2.append(dat[i].skew())
    for n in range(4):
        if abs(temp1[n])>abs(temp2[n]):
            skewFeature.append(NetF[n])


    # In[9]:


    for i in skewFeature:
        if (data[i].min()) > 0 :
            data[i] = data[i].map(lambda k: np.log(k)if k > 0 else 0)
        #Checking
        if (data[i].min()) <= 0 :
            data[i] = data[i].map(lambda x: -np.power(-x, 1./3) if x < 0 else np.power(x, 1./3)) 


    # In[10]:


    #Outliers and removing them

    Ps = data.quantile(0.25)
    Ps1 = data.quantile(0.75)
    result = Ps1 - Ps

    data = data[~((data < (Ps - 2 * result)) |(data > (Ps1 + 2 * result))).any(axis=1)]
    data.shape
    print("Outliers and skewness removed")




    #Scalling the dataset

    from sklearn.preprocessing import StandardScaler,scale
    #we create an object of the class StandardScaler
    sc = StandardScaler()

    col_to_scale = NetF
    data[col_to_scale] = sc.fit_transform(data[col_to_scale])
    data.head()
    print("Dataset Scaled")



    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(data, train_size = 0.8, test_size = 0.2, random_state = 0 )
    #Splitting it based on that cofficient 
    y_train = data_train.loc[:,data_train.columns == 'GT Turbine decay state coefficient.']
    X_train = data_train.loc[:, data_train.columns != 'GT Turbine decay state coefficient.']
    y_test = data_test.loc[:,data_test.columns == 'GT Turbine decay state coefficient.']
    X_test = data_test.loc[:, data_test.columns != 'GT Turbine decay state coefficient.']


    # In[45]:


    X_train.head()


    # <b> GT Turbine decay state coefficient minimizing ‘rmse’ loss function </b> <br>
    # Root mean squared error function

    # In[46]:


    from keras import backend as K
    def root_mean_squared_error(y_act, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_act), axis=-1)) 


    # In[47]:
    print("Model training:")
    #Model
    #Input layer (17 dims)
    # Two dense layers(nodes = 64)
    #Output Layer
    from keras.models import Sequential
    from keras.layers import Dense
    model1 = Sequential()
    #input layer
    model1.add(Dense(input_dim = 17, units = 256, kernel_initializer='normal', activation='relu'))
    #1 Dense Layers
    model1.add(Dense(64,activation = 'relu', kernel_initializer='normal'))
    model1.add(Dense(64,activation = 'relu', kernel_initializer='normal'))
    #Output layer
    model1.add(Dense(1, kernel_initializer='normal' ))
    model1.compile(optimizer='adam', loss=root_mean_squared_error)
    model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size = 50,verbose = 2)


    # In[48]:


    #Predicting
    y_pred = model1.predict(X_test)
    #R2 score 
    from sklearn.metrics import r2_score 
    print("R2 score is: ",r2_score(y_test, y_pred))


    # In[49]:


    #Residual
    resid = y_test - y_pred
    #Durbin wayson test for autocorrelation among error terms, close to 2, so error terms
    # are almost indepedent of each other
    from statsmodels.stats.stattools import durbin_watson
    print("Durbin watson auto correlation score:",durbin_watson(resid))
    return model1
    


#To test the compressor decay cofficient model
#model = model_compressor("D://Data/propulsion.csv")
#model.predict(X)

#To test the turbo model
#model1 = model_turbo("D://Data/propulsion.csv")
#model1.predict(X)


