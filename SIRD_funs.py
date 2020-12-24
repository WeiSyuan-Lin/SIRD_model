#!/usr/bin/env python
# coding: utf-8

# In[21]:


import requests
import pandas as pd
import torch.optim as optim
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


# In[17]:


Country=['Italy','Germany','United Kingdom','Spain','US','France','China','Belgium','Egypt','Kenya']
population_N=[60.36*1000000, 83.02*1000000 ,66.56*1000000 ,46.94*1000000 ,327.2*1000000 ,66.99*1000000, 1386*1000000 ,11.46*1000000,10*1000000,4.8*1000000]


# In[6]:


def input_all_data():
    DATA=[]
    for i in range(27,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,30):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/02-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,31):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/05-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,31):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/06-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/07-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,31):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/09-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,32):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/10-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    for i in range(1,31):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/11-'+str(i)+'-2020.csv'
        res = requests.get(url, allow_redirects=True)
        with open('covid.csv','wb') as file:
            file.write(res.content)
        DATA.append(pd.read_csv('covid.csv'))
    return DATA


# In[7]:


#DATA=input_all_data()


# In[8]:


def Data_of_Country(num,DATA): 
    Data_Country=[]
    for i in range(len(DATA)):
        if 'Country/Region' in DATA[i]:
            fliter = (DATA[i]['Country/Region']==Country[num])
            if len(DATA[i][fliter])>0:
                Data_Country.append(DATA[i][fliter])
        elif 'Country_Region' in DATA[i]:
            fliter = (DATA[i]['Country_Region']==Country[num])
            if len(DATA[i][fliter])>0:
                Data_Country.append(DATA[i][fliter])
    return Data_Country


# In[9]:


def take_SIRD_data(Data_Country,number):
    num_of_Data_Country=len(Data_Country)
    N=population_N[number]
    St=[]
    for i in range(num_of_Data_Country):
        St.append(N-Data_Country[i]['Confirmed'].sum()-Data_Country[i]['Deaths'].sum()-Data_Country[i]['Recovered'].sum())
    St=np.array(St)
    It=[]
    for i in range(num_of_Data_Country):
        It.append(Data_Country[i]['Confirmed'].sum())
    It=np.array(It)
    Rt=[]
    for i in range(num_of_Data_Country):
        Rt.append(Data_Country[i]['Recovered'].sum())
    Rt=np.array(Rt)
    Dt=[]
    for i in range(num_of_Data_Country):
        Dt.append(Data_Country[i]['Deaths'].sum())
    Dt=np.array(Dt)
    return St,It,Rt,Dt


# In[22]:


def SIRD_data(DATA,Countryname):
    It=[]
    Rt=[]
    Dt=[]
    for i in range(len(DATA)):
        if 'Country/Region' in DATA[i]:
            fliter = (DATA[i]['Country/Region']==Countryname)
            if len(DATA[i][fliter])>0:
                It.append(DATA[i][fliter]['Confirmed'].sum())
                Rt.append(DATA[i][fliter]['Recovered'].sum())
                Dt.append(DATA[i][fliter]['Deaths'].sum())
            else:
                It.append(0)
                Rt.append(0)
                Dt.append(0)
        elif 'Country_Region' in DATA[i]:
            fliter = (DATA[i]['Country_Region']==Countryname)
            if len(DATA[i][fliter])>0:
                It.append(DATA[i][fliter]['Confirmed'].sum())
                Rt.append(DATA[i][fliter]['Recovered'].sum())
                Dt.append(DATA[i][fliter]['Deaths'].sum())
            else:
                It.append(0)
                Rt.append(0)
                Dt.append(0)
    It=np.array(It)
    Rt=np.array(Rt)
    Dt=np.array(Dt)
    return It,Rt,Dt


# In[11]:


import numpy as np
from scipy.optimize import minimize

def objective(x):
    St_pred=[]
    It_pred=[]
    Dt_pred=[]
    Rt_pred=[]
  #參數 dt 單位天數
    dt=1
    eps=0.0000001
    beta=x[0]
    gamma=x[1]
    mu=x[2]
    #N=population_N[num]
  
  #藉由finite difference method 做出下一個時間點的預測
   
    St_pred.append(St_now*(1-dt*beta*It_now/N))
    It_pred.append(It_now*(1+dt*(beta*St_now/N)-dt*(mu+gamma)))
    Rt_pred.append(Rt_now+(dt*gamma*It_now))
    Dt_pred.append(Dt_now+(dt*mu*It_now))
  
  # loss function 只取用論文中使用的loss function的前兩項
    loss=(100/len(It_pred))*(((It_true-It_pred)**2).sum()+((Dt_true-Dt_pred)**2).sum())                                                                                                                
    return loss


# In[12]:


def find_parameter(num,population_N):
  # initial guesses
    beta = 0.1
    gamma = 0.1
    mu = 0.001
    N=population_N[num]

    x0 = np.zeros(3)
    x0[0] = beta
    x0[1] = gamma
    x0[2] = mu
    b = (0.0,1.0)
    bnds = (b, b, b)
    
    Sol=minimize(objective,x0,bounds=bnds,options= {'disp':True,'maxiter':6000000})
    return Sol.x


# In[16]:


def predict(beta,gamma,mu,N,St_now,It_now,Rt_now,Dt_now):
    dt=1
    St_pred=St_now*(1-dt*beta*It_now/N)
    It_pred=It_now*(1+dt*(beta*St_now/N)-(mu+gamma))
    Rt_pred=Rt_now+(dt*gamma*It_now)
    Dt_pred=Dt_now+(dt*mu*It_now)
    return St_pred,It_pred,Rt_pred,Dt_pred


# In[45]:


#Country=['Netherlands','United Kingdom','France','Belgium','Spain','Italy','Germany','US','Egypt','Kenya','Japan','Austria','Qatar']


# In[ ]:





# In[46]:


def predict_train(DATA,Country,day):
    Death_train=[]
    Death_trainlabel=[]
    Infected_train=[]
    Infected_trainlabel=[]
    for j in range(5):
        [It,Rt,Dt]=SIRD_data(DATA,Country[j])
        It=It.reshape(len(It),1)
        Rt=Rt.reshape(len(Rt),1)
        Dt=Dt.reshape(len(Dt),1)
        IRD_data_pair=np.hstack([It,Rt,Dt])
        for i in range(0,round(len(IRD_data_pair)*0.6)-29-day):
            I_mean=It[i:i+30].mean()
            I_std=It[i:i+30].std()+1
            R_mean=Rt[i:i+30].mean()
            R_std=Rt[i:i+30].std()+1
            D_mean=Dt[i:i+30].mean()
            D_std=Dt[i:i+30].std()+1
            I_Mean=It.mean()
            I_Std=It.std()
            R_Mean=Rt.mean()
            R_Std=Rt.std()
            D_Mean=Dt.mean()
            D_Std=Dt.std() 
    
            Infected_train.append(np.hstack([((It[i:i+30]-I_mean)/I_std),((Dt[i:i+30]-D_mean)/D_std)]).reshape(2*30,1).tolist())
            Infected_trainlabel.append((It[i+29+day]-I_mean)/I_std)

            Death_train.append(np.hstack([((It[i:i+30]-I_mean)/I_std),((Dt[i:i+30]-D_mean)/D_std)]).reshape(2*30,1).tolist())
            Death_trainlabel.append((Dt[i+29+day]-D_mean)/D_std)


    Death_train=np.array(Death_train).reshape(len(Death_train),60)
    Death_trainlabel=np.array(Death_trainlabel).reshape(-1,1)

    Infected_train=np.array(Infected_train).reshape(len(Infected_train),60)
    Infected_trainlabel=np.array(Infected_trainlabel).reshape(-1,1)
        
    from sklearn.linear_model import LinearRegression
        
    reg_D = LinearRegression().fit(Death_train,Death_trainlabel)
    reg_I = LinearRegression().fit(Infected_train,Infected_trainlabel)

    data_dmatrix_D = xgb.DMatrix(data=Death_train,label=Death_trainlabel)
    data_dmatrix_I = xgb.DMatrix(data=Infected_train,label=Infected_trainlabel)
    param = {'max_depth':10, 'eta':0.5,'booster':'gbtree'}
    num_round = 100
    bst_D = xgb.train(param,data_dmatrix_D,num_round)
    bst_I = xgb.train(param,data_dmatrix_I,num_round)

    return bst_D,bst_I,reg_D,reg_I


# In[47]:





# In[55]:


def predict_test(DATA,Countryname,bst_D,bst_I,reg_D,reg_I,day):
    Death_test=[]
    Death_testlabel=[]
    Infected_test=[]
    Infected_testlabel=[]

    Infected_true=[]
    Death_true=[]
    I_Means=[]
    I_Stds=[]
    D_Means=[]
    D_Stds=[]

    [It,Rt,Dt]=SIRD_data(DATA,Countryname)
    It=It.reshape(len(It),1)
    Rt=Rt.reshape(len(Rt),1)
    Dt=Dt.reshape(len(Dt),1)
    IRD_data_pair=np.hstack([It,Rt,Dt])
    for i in range(len(IRD_data_pair)-29-day):
        I_mean=It[i:i+30].mean()
        I_std=It[i:i+30].std()+1
        R_mean=Rt[i:i+30].mean()
        R_std=Rt[i:i+30].std()+1
        D_mean=Dt[i:i+30].mean()
        D_std=Dt[i:i+30].std()+1
        
        Infected_test.append(np.hstack([((It[i:i+30]-I_mean)/I_std),((Dt[i:i+30]-D_mean)/D_std)]).tolist())
        Death_test.append(np.hstack([((It[i:i+30]-I_mean)/I_std),((Dt[i:i+30]-D_mean)/D_std)]).tolist())
        
        Infected_true.append((It[i+29+day]-I_mean)/I_std)
        Death_true.append((Dt[i+29+day]-D_mean)/D_std)
        
        I_Means.append(I_mean)
        I_Stds.append(I_std)
        D_Means.append(D_mean)
        D_Stds.append(D_std)
    
    Death_test=np.array(Death_test).reshape(len(Death_test),60)
    Infected_test=np.array(Infected_test).reshape(len(Infected_test),60)
    I_Means=np.array(I_Means).reshape(len(I_Means),1)
    I_Stds=np.array(I_Stds).reshape(len(I_Stds),1)
    D_Means=np.array(D_Means).reshape(len(D_Means),1)
    D_Stds=np.array(D_Stds).reshape(len(D_Stds),1)
    
    Death_true=np.array(Death_true).reshape(-1,1)
    Infected_true=np.array(Infected_true).reshape(-1,1)

    preds_D = bst_D.predict(xgb.DMatrix(Death_test))
    preds_I = bst_I.predict(xgb.DMatrix(Infected_test))


    plt.figure()
    dat=np.linspace(0,len(Dt),len(Dt))

    XGD=np.vstack([np.zeros(29+day).reshape(-1,1),preds_D.reshape(-1,1)*D_Stds+D_Means])
    res_D=np.vstack([np.zeros(29+day).reshape(-1,1),reg_D.predict(Death_test)*D_Stds+D_Means])

    plt.plot(dat,XGD,label='XGboost') #預測人數
    plt.plot(dat,res_D,label='linear regression')

    plt.plot(dat,Dt,label='true')#實際人數


    plt.title('D of '+Countryname)
    plt.xlabel('day')
    plt.ylabel('(D)')
    plt.legend()


    plt.figure()
    dat=np.linspace(0,len(It),len(It))

    XGI=np.vstack([np.zeros(29+day).reshape(-1,1),(preds_I.reshape(-1,1))*I_Stds+I_Means])
    res_I=np.vstack([np.zeros(29+day).reshape(-1,1),(reg_I.predict(Infected_test).reshape(-1,1))*I_Stds+I_Means])

    plt.plot(dat,XGI,label='XGboost') #預測人數
    plt.plot(dat,res_I,label='linear regression')
    plt.plot(dat,It,label='true')#實際人數

    plt.title('I of '+Countryname)
    plt.xlabel('day')
    plt.ylabel('(I)')
    plt.legend()
    print('預測'+str(day)+'天後的結果')
    print('')
    print('XGboost預測死亡最大誤差',np.linalg.norm((XGD-Dt),np.inf))
    print('XGboost平均死亡誤差和標準差',[(np.abs(XGD-Dt)).mean(),(np.abs(XGD-Dt)).std()])

    print('Linear resgression預測死亡最大誤差',np.linalg.norm((res_D-Dt),np.inf))
    print('Linear resgression平均死亡誤差和標準差',[(np.abs(res_D-Dt)).mean(),(np.abs(res_D-Dt)).std()])
    print('')
    print('XGboost預測感染最大誤差',np.linalg.norm((XGI-It),np.inf))

    print('XGboost平均感染誤差和標準差',[(np.abs(XGI-It)).mean(),(np.abs(XGI-It)).std()])
    print('Linear resgression預測感染最大誤差',np.linalg.norm((res_I-It),np.inf))
    print('Linear resgression平均死亡誤差和標準差',[(np.abs(res_I-It)).mean(),(np.abs(res_I-It)).std()])
    print('')


# In[64]:


#day=10
#[bst_D,bst_I,reg_D,reg_I]=predict_train(DATA,Country,day)


# In[65]:


#predict_test(DATA,'United Kingdom',bst_D,bst_I,reg_D,reg_I,day)


# In[ ]:





# In[ ]:




