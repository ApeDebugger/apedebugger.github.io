#Importing required modules
import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt

'''Before proceeding format the catchment file as following: 
    column1: Date, 
    column2: Rainfall,
    column3: ET or blank (because of physical models)
    column4: Runoff'''

#CHOOSE CATCHMENT FILE (godaveri.csv or jardine.csv)
datafile = 'jardine.csv'
#Choose training data length (3000 for godaveri.csv and 3500 for jardine.csv)
ntrain = 3500
#Choose no. of lagged steps backward (n_in) and forward (n_out) for data preparation
n_in = 2
n_out = 1
#Choose no. of epochs (or iterations) for modelling
epoch = 50
#Choose validation(0), testing(1) or cross validation (2)
#test = 2 is for selection of hidden layer neurons by cross-validation
test = 1

#Choose validation data length (choose ~500 for both Godaveri and Jardine)
nval = 498
#Choose no. of hidden layer neurons
nneuron = 11


#Read the data
data = pd.read_csv("C:\\Users\\ShubhM\\Desktop\\CE491A\\Project Files\\Data for project\\%s" %(datafile), 
                   sep = ",", index_col = 0, usecols =[0,1,3], 
                   parse_dates = True)
data[data == -999] = pd.np.nan
array = data.values


#Checking Statistics of data
#print(data.describe())
#print(data.isnull().sum())


#Adding lagtime series of Q and R 
#Method and functions for converting data to the time-lag form
def lagvariable(data, n_in, n_out):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a DataFrame.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        
    Returns:
        Pandas DataFrame of lagged series of an input variable.
        """
    colname = data.columns[0]
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (colname, i))]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % (colname))]
        else:
            names += [('%s(t+%d)' % (colname, i))]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    return agg

def lagdata(data, n_in, n_out):
    """
    Arguments: 
        Read above method's comments for understanding the terms 
        of this method.
        n_vars: No. of variables
    
    Returns:
        Pandas DataFrame of lagged series of all variables.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    lag_data = []
    for i in range(n_vars):
        data1 = pd.DataFrame(data[data.columns[i]])           
        lag_data.append(lagvariable(data1, n_in, n_out))
    lag_data = concat(lag_data, axis=1)
    return lag_data


#Adding logQ variable with its density plot
#logQ = np.log(array[:,1])
#data['logQ'] = logQ


#Time Series of Q and P
'''data['Q'].plot(title='Daily Runoff Time Series', 
      sharex = False, figsize = (30,10), color = 'blue')
plt.legend(loc='best')
plt.show()
data['R(mm)'].plot(title='Daily Rainfall Time Series', 
      sharex = False, figsize = (30,10), color = 'red')
plt.legend(loc='best')
plt.show()
data['ET(mm)'].plot(title='Daily ET Time Series', 
      sharex = False, figsize = (30,10), color = 'green')
plt.legend(loc='best')
plt.show()'''


#Histogram Matrix
'''plt.hist(array[:,0], 200, normed=1, facecolor='g', alpha=0.75)
plt.xlabel(data.columns[0])
plt.ylabel('Probability')
plt.title('Histogram of R')
plt.axis([0, 40, 0, 0.03])
plt.show()'''


#Scatter plots of different attributes
#plt.scatter(array[:,0], array[:,2], c="b", alpha=0.5,label="RvsQ")
'''fig, ax = plt.subplots(1,3,figsize=(9,3))
ax[0,0].scatter(array[:,0], array[:,1], color = "blue")
ax[0,1].scatter(array[:,0], array[:,2], color = "green")
ax[0,2].scatter(array[:,1], array[:,2], color = "red")
ax[0,0].set_title("RvsET")
ax[0,1].set_title("RvsQ")
ax[0,2].set_title("ETvsQ")
ax[0,0].set_xlabel("R")
ax[0,1].set_xlabel("R")
ax[0,2].set_xlabel("ET")
ax[0,0].set_ylabel("ET")
ax[0,1].set_ylabel("Q")
ax[0,2].set_ylabel("Q")
#fig.subplots_adjust(wspace=0.5)
fig.suptitle("Scatterplots")
plt.show()'''


#Scatterplot matrix
'''from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()'''


#Auto & Partial Correlation Graphs
from statsmodels.tsa.stattools import acf, pacf
def autopartcorr(data, c):
    data = data.dropna(axis = 0)
    f = pd.DataFrame(acf(data[c]), columns = [c])
    g = pd.DataFrame(pacf(data[c]), columns = [c])
    index = f.iloc[1:20,:].index.values
    f1 = f.iloc[1:20,:].values
    g1 = g.iloc[1:20,:].values
    plt.bar(index, f1, align='center', color = 'blue', alpha=0.7, label='ACF')
    plt.bar(index, g1, align='center', color = 'green', label='PACF')
    plt.show()
    return f, g


#Uncomment below cell to calculate auto-corr and partial auto-corr functions values
'''c = data.columns[1]
acf1, pacf1 = autopartcorr(data, c)'''


#Correlation heatmap for both data and lag_data
'''import seaborn as sns
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()'''


#Dividing train and testing data
#Dropping Missing Values
data = data.dropna(axis = 0)

#method for Dividing Training & Testing data
def split(data, ntrain):
    n = data.shape[0]
    traindata = data.iloc[0:ntrain,:] 
    ntest = n - ntrain
    testdata = data.iloc[ntrain+1:(ntrain+ntest),:]
    return traindata, testdata

#method for feature Scaling 
from sklearn.preprocessing import MinMaxScaler
def scaledata(data, ntrain):
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    train, test = split(data, ntrain)
    return train, test

#method for converting train, test into lagged data
def finallag(data, ntrain, n_in, n_out):
    train, test = scaledata(data, ntrain)
    lag_train = lagdata(train, n_in, n_out)
    lag_test = lagdata(test, n_in, n_out)
    return lag_train, lag_test

#method for input and output split for train and test data
def XYsplit(data, ntrain, n_in, n_out):
    train, test = finallag(data, ntrain, n_in, n_out)
    train = train.dropna(axis = 0)
    test = test.dropna(axis = 0)
    Xtrain = train.iloc[:,:-1].values
    Ytrain = train.iloc[:,-1].values
    Xtest = test.iloc[:,:-1].values
    Ytest = test.iloc[:,-1].values
    return Xtrain, Xtest, Ytrain, Ytest

#choosing no. of training points
ntrain = ntrain
#Comparing statistical properties of train and test data
def statcompare(data, ntrain):
    traindata, testdata = scaledata(data, ntrain)
    print(traindata.describe())
    print(testdata.describe())

#statcompare(data, ntrain)
#Change ntrain and do necessary changes in training and testing split
#for better predictions 

#Uncomment below to save .csv file of final lagged data and caculate correlation values
'''lag_data = []
lagtrain, lagtest = finallag(data, ntrain, n_in, n_out)
lagtrain = lagtrain.dropna(axis = 0)
lagtest = lagtest.dropna(axis = 0)
lag_data.append(lagtrain)
lag_data.append(lagtest)
lagdata1 = concat(lag_data, axis = 0)
lagdata1.to_csv("C:\\Users\\ShubhM\\Desktop\\CE491A\\Project Files\\Data for project\\lagjardine.csv", sep=',')
#Correlation Matrix
corrm = lagdata1.corr()'''


#Defining Correlation coeffcient for model evaluation
from math import sqrt
def corr(y_true, y_pred):
    '''
        y_true -> true output array
        y_pred -> predicted output array
        Qobar -> mean of true outputs
        Qmbar -> mean of predicted outputs
    '''
    Qobar = np.mean(y_true)
    Qmbar = np.mean(y_pred)
    n = len(y_true)
    covar = 0
    varQo, varQm = 0, 0
    for i in range(n):
        covar += (y_true[i]-Qobar)*(y_pred[i]-Qmbar)
        varQo += ((y_true[i]-Qobar)*(y_true[i]-Qobar))
        varQm += ((y_pred[i]-Qmbar)*(y_pred[i]-Qmbar))
    corr = covar/sqrt(varQo*varQm)
    return corr

#ANN Models' Evaluation Metrics function
def evalmet(Qo, Qm):
    '''
        mbe -> Mean Bias Error
        mae -> Mean Absolute Error
        TSX -> Threshold Statistics below X%
        mare -> Mean Absolute Relative Error
        mape% -> Mean Absolute % Error
        E -> Nash-Sutcliffe Efficiency, value=-infinity to 1
        MF% -> relative % error in maximum value of Qo
    '''
    n = len(Qo)
    be, ae, are, arpe = 0, [], [], []
    Qobar = np.mean(Qo)
    for i in range(n):
        be += (Qo[i]-Qm[i])/Qobar
        ae.append(abs(Qo[i]-Qm[i]))
        are.append(abs((Qo[i]-Qm[i])/Qo[i]))
        arpe.append(abs(((Qo[i]-Qm[i])*100)/Qo[i]))
    mbe = be*100/float(n)
    mae = sum(ae)/float(n)
    arpe = np.array(arpe)
    TS1 = (sum(arpe<1))*100/n
    TS25 = (sum(arpe<25))*100/n
    TS50 = (sum(arpe<50))*100/n
    TS100 = (sum(arpe<100))*100/n
    mare = sum(are)/float(n)
    mape = mare*100
    e1, e2 = 0, 0
    for i in range(n):
        e1 += (Qo[i]-Qobar)*(Qo[i]-Qobar)
        e2 += (Qo[i]-Qm[i])*(Qo[i]-Qm[i])
    E = (e1-e2)/e1
    Qo = np.array(Qo)
    MF = 100*((Qm[np.argmax(Qo)]-max(Qo))/max(Qo))
    results = []
    metrics = ['mbe', 'mae', 'mare', 'mape', 'TS1', 'TS25', 'TS50', 'TS100', 'E', 'MF']
    metrics1 = [mbe, mae, mare, mape, TS1, TS25, TS50, TS100, E, MF]
    j = 0
    for i in metrics:
        i = pd.DataFrame(metrics1[j], columns = [i])
        results.append(i)
        j += 1
    results = concat(results, axis=1)
    return results


#Design ANN MODEL
seed = 7
np.random.seed(seed)
# define a base MLPNN model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
def baseline_model(n_hidneurons, n_input):
    # create model
    model = Sequential()
    model.add(Dense(n_hidneurons, input_dim = n_input))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model


def validatemodel(Xtrain, Ytrain, Xtest, Ytest, nval, model, optimizer, test, epoch):
    tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs',
                                         write_graph=True)
    
    #Compile model
    model.compile(loss='mse', optimizer=optimizer)
    
    if test==2:
        #For cross validation modelling (hidden layer neurons selection)
        #Cross-Validation Split
        tscv = TimeSeriesSplit(n_splits=3)

        trainrmse = []
        valrmse = []
        corrtrain = []
        corrval = []
        for train, test in tscv.split(Xtrain, Ytrain):
            # Fit the model
            model.fit(Xtrain[train], Ytrain[train], epochs=epoch, 
                      batch_size=10, verbose=1, callbacks=[tbCallBack])
            # evaluate the model
            model.evaluate(Xtrain[test], Ytrain[test], verbose=0)
            #Preciting the output values for train and val data
            yhattrain = model.predict(Xtrain[train])
            yhatval = model.predict(Xtrain[test])
            #Calculating Correlation Coefficient for train and val data
            corrtrain.append(corr(Ytrain[train], yhattrain))
            corrval.append(corr(Ytrain[test], yhatval))
            #Calculating Root Mean Square Error for train and val data
            trainrmse.append(sqrt(mean_squared_error(Ytrain[train], yhattrain)))
            valrmse.append(sqrt(mean_squared_error(Ytrain[test], yhatval)))
        return trainrmse, valrmse, corrtrain, corrval
    else:
        #For validation and testing modelling (hidden layer neurons selection)
        ntrain = Xtrain.shape[0]
        ntrain = ntrain-nval
        Xval = Xtrain[(ntrain+1):(ntrain+nval)]
        Yval = Ytrain[(ntrain+1):(ntrain+nval)]
        Xtrain = Xtrain[0:ntrain]
        Ytrain = Ytrain[0:ntrain]
        #Fit the model
        History = model.fit(Xtrain, Ytrain, epochs=epoch, batch_size=10, 
                            validation_data=(Xtest, Ytest),
                            verbose=1, callbacks=[tbCallBack])
        epocherr = History.history
        #Predicting the output values for train and val data
        yhattrain = model.predict(Xtrain) 
        yhatval = model.predict(Xval)
        #Calculating Correlation Coefficient for train and val data
        corrtrain = corr(Ytrain, yhattrain)
        corrval = corr(Yval, yhatval)
        #Calculating Root Mean Square Error for train and val data
        trainrmse = (sqrt(mean_squared_error(Ytrain, yhattrain)))
        valrmse = (sqrt(mean_squared_error(Yval, yhatval)))
    
        if test==0:    
            return epocherr, Ytrain, Yval, yhattrain, yhatval, trainrmse, valrmse, corrtrain, corrval
        else:
            return Ytrain, yhattrain, trainrmse, corrtrain
    

def modeltesting(data, ntrain, nval, n_in, n_out, n_hidneurons, test, epoch):
    #Split data here
    Xtrain, Xtest, Ytrain, Ytest = XYsplit(data, ntrain, n_in, n_out)
    n_input = Xtrain.shape[1]
    
    #Choose model
    model = baseline_model(n_hidneurons, n_input)
    
    #Model summary and optimization function selection
    model.summary()
    #choose optimizer
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if test==0:
        #call simulation method
        epocherr, Ytrain, Yval, yhattrain, yhatval, trainrmse, valrmse, corrtrain1, corrval = validatemodel(Xtrain, 
                                                               Ytrain,
                                                               Xtest,
                                                               Ytest,
                                                               nval,
                                                               model, 
                                                               adam,
                                                               test,
                                                               epoch)
        trainrmse = np.mean(trainrmse)
        valrmse = np.mean(valrmse)
        corrval = np.mean(corrval)
        corrtrain1 = np.mean(corrtrain1)
        return epocherr, Ytrain, Yval, yhattrain, yhatval, corrtrain1, corrval, trainrmse, valrmse 

    elif test==1:
        #call simulation method
        Ytrain, yhattrain, trainrmse, corrtrain1 = validatemodel(Xtrain, 
                                                               Ytrain,
                                                               Xtest,
                                                               Ytest,
                                                               nval,
                                                               model, 
                                                               adam,
                                                               test,
                                                               epoch)
        #Evaluating the model for testdata (Only use when a best model 
                                        #is selected)
        yhattest = model.predict(Xtest, batch_size=1)
        corrtest = corr(Ytest, yhattest)
        rmsetest = sqrt(mean_squared_error(Ytest, yhattest))
        weight = model.get_weights()
        return Ytrain, Ytest, yhattrain, yhattest, corrtrain1, corrtest, trainrmse, rmsetest, weight 
    else: 
        trainrmse, valrmse, corrtrain1, corrval = validatemodel(Xtrain, 
                                                               Ytrain,
                                                               Xtest,
                                                               Ytest,
                                                               nval,
                                                               model, 
                                                               adam,
                                                               test,
                                                               epoch)
        trainrmse = np.mean(trainrmse)
        valrmse = np.mean(valrmse)
        corrval = np.mean(corrval)
        corrtrain1 = np.mean(corrtrain1)
        return trainrmse, valrmse, corrtrain1, corrval


def HiddenNeuronsSelection(data, n_in, n_out, ntrain, test, epoch):
    nval = 0
    ntrainrmse = []
    nvalrmse = []
    ncorrtrain = []
    ncorrval = []
    for nneuron in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
        cvtrainrmse, cvvalrmse, corrtrain, corrval = modeltesting(data,
                                                                  ntrain,
                                                                  nval,
                                                                  n_in,
                                                                  n_out,
                                                                  nneuron,
                                                                  test,
                                                                  epoch)
        ntrainrmse.append(cvtrainrmse)
        nvalrmse.append(cvvalrmse)
        ncorrtrain.append(corrtrain)
        ncorrval.append(corrval)
    return ntrainrmse, nvalrmse, ncorrtrain, ncorrval


'''
    Explaination of variables mentioned below:
        trainrmse
'''

if test==2:
    rmsetrain, rmseval, corrtrain, corrval = HiddenNeuronsSelection(data, 
                                                                    n_in, 
                                                                    n_out, 
                                                                    ntrain,
                                                                    test,
                                                                    epoch)

    #Choose number of hidden layer neurons with best results from above metrics
    nneuron = nneuron
#Model Evaluation
elif test==0:
    epocherr, Ytrain, Yval, yhattrain, yhatval, corrtrain, corrval, rmsetrain, rmseval = modeltesting(data, 
                                                           ntrain, 
                                                           nval,
                                                           n_in, 
                                                           n_out, 
                                                           nneuron,
                                                           test,
                                                           epoch)
    #Error metrics for training & validation
    trainres = evalmet(Ytrain, yhattrain)
    valres = evalmet(Yval, yhatval)
elif test==1:
    Ytrain, Ytest, yhattrain, yhattest, corrtrain, corrtest, rmsetrain, rmsetest, weights = modeltesting(data, 
                                                           ntrain, 
                                                           nval,
                                                           n_in, 
                                                           n_out, 
                                                           nneuron,
                                                           test,
                                                           epoch)
    #Error Metrics for training
    trainres = evalmet(Ytrain, yhattrain)
    trainres['R'] = corrtrain
    trainres['RMSE'] = rmsetrain
    #Final Evaluation for best model
    #Evaluating the model for testdata (Only use when a best model is selected)
    testres = evalmet(Ytest, yhattest)
    testres['R'] = corrtest
    testres['RMSE'] = rmsetest
else: 
    print("Looks like you have failed your test of comprehensibility. Please kindly change the value! Oh I am sorry! it must have been tough to deciper the previous enigmatic text, let me make it easier for you: Fucking choose ")  


# Plot scatterplots for predicted vs observed runoff
'''plt.scatter(Ytrain, yhattrain, c="b", alpha=0.5,label="Predicted vs True Output")
plt.show()
plt.scatter(Ytest, yhattest, c="b", alpha=0.5,label="Predicted vs True Output")
plt.show()'''
#train
'''plt.plot(yhattrain, label='Predicted Output')
plt.plot(Ytrain, label='True Output')
plt.legend(loc='best')
plt.show()'''
#test
'''plt.plot(yhattest, label='Predicted Output')
plt.plot(Ytest, label='True Output')
plt.legend(loc='best')
plt.show()'''