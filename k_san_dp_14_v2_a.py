import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import scipy as sp
#from skll import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD,Adam,rmsprop,Adagrad,Adadelta,Adamax

np.random.seed(123)

# encode class values as integers
def one_hot_encod(df,cat_list,num_list):
    dummy_x = df[num_list].as_matrix()
    for i in cat_list:
        encoder = LabelEncoder()
        encoder.fit(df[i])
        encoded_x = encoder.transform(df[i])
        dummy_y = np_utils.to_categorical(encoded_x)
        dummy_x = np.hstack((dummy_x,dummy_y))
    return(np.array(dummy_x))

wdir='/Users/kjosephkujur/Downloads/k_san/'
#execfile(wdir +"kaggle_bim_prd_clus.py")

#Loading data
df_train = pd.read_csv(wdir + 'train_ver2.csv')
df_sub = pd.read_csv(wdir + 'test_ver2.csv')

# All Classes(products) to predict
products = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

#Data with purchase only for a particular month
def users_fecha_dato(in_data,fecha_date):
    temp = pd.DataFrame(in_data.ix[in_data.fecha_dato==fecha_date,:].groupby(['ncodpers'])[products].sum())
    temp.reset_index(level=0, inplace=True)
    temp = temp.fillna(0)
    out_data = pd.melt(temp, id_vars="ncodpers",var_name="Var",value_name="Val1")
    out_data = out_data.ix[out_data.Val1 > 0,:]
    return(out_data)

def monthly_new_prod_users(df,fecha_date_prev,fecha_date_cur):
    df_train_prev = users_fecha_dato(df,fecha_date_prev)
    df_train_cur = users_fecha_dato(df,fecha_date_cur)
    users_to_take = pd.merge(df_train_cur[['ncodpers','Var']],df_train_prev,on=['ncodpers','Var'],how='left')
    users_to_take = users_to_take.ix[users_to_take.Val1.isnull(),0:2] #this has users who only bought something new Jun'2015
    Target_y = pd.get_dummies(users_to_take['Var'])
    for i in list(set(products) - set(users_to_take.Var)):
        Target_y[i] = 0
    Target_y['ncodpers'] = users_to_take['ncodpers']
    Target_y = Target_y.fillna(0)
    return(Target_y)
    
user_new_prod_0416 = monthly_new_prod_users(df_train,'2016-03-28','2016-04-28')
user_new_prod_0615 = monthly_new_prod_users(df_train,'2015-05-28','2015-06-28')

# Creating training data set                                 
def training_set(df_train,df,date):
    df_train_new = df_train.ix[(df_train['fecha_alta'] < date) & (df_train['fecha_dato'] == date) & (df_train['ncodpers'].isin(np.unique(df.ncodpers))),:]
    df_train_new[products] = df_train_new[products].fillna(0)
    df_train_new[products] = df_train_new[products].astype(int)
    return(df_train_new)

df_train_new_0416 = training_set(df_train,user_new_prod_0416,'2016-04-28')
df_train_new_0615 = training_set(df_train,user_new_prod_0615,'2015-06-28')
    
#df_train_new.describe()
#Variables not considered -
    #ind_nuevo    
    #tipodom
    #cod_prov
    #pais_residencia
    #ult_fec_cli_1t
    #fecha_dato
    #fecha_alta
    #indrel_1mes
    #conyuemp
    #canal_entrada
    #antiguedad

# Data Preparation for demographics (non -transactional)
def data_prep_dmg(df):
    df['antiguedad'] = df['antiguedad'].convert_objects(convert_numeric=True)
    df['age'] = df['age'].convert_objects(convert_numeric=True)
    df['renta'] = df['renta'].convert_objects(convert_numeric=True)
    df['indrel_1mes'] = df['indrel_1mes'].convert_objects(convert_numeric=True)
    
    df['tenure'] = ((pd.to_datetime(df['fecha_dato']) - pd.to_datetime(df['fecha_alta'])).dt.days).astype(int)/30
    df['tenure_grp1'] = 'tI'
    df.loc[(df['tenure'] <= 1),'tenure_grp1'] = 'tA1'
    df.loc[(df['tenure'] > 1) & (df['tenure'] <= 3),'tenure_grp1'] = 'tA2'
    df.loc[(df['tenure'] > 3) & (df['tenure'] <= 6),'tenure_grp1'] = 'tA3'
    df.loc[(df['tenure'] > 6) & (df['tenure'] <= 12),'tenure_grp1'] = 'tB'
    df.loc[(df['tenure'] > 12) & (df['tenure'] <= 24),'tenure_grp1'] = 'tC'
    df.loc[(df['tenure'] > 24) & (df['tenure'] <= 36),'tenure_grp1'] = 'tD'
    df.loc[(df['tenure'] > 36) & (df['tenure'] <= 60),'tenure_grp1'] = 'tE'
    df.loc[(df['tenure'] > 60) & (df['tenure'] <= 96),'tenure_grp1'] = 'tF'
    df.loc[(df['tenure'] > 96) & (df['tenure'] <= 144),'tenure_grp1'] = 'tG'
    df.loc[(df['tenure'] > 144) & (df['tenure'] <= 192),'tenure_grp1'] = 'tH'
    
    df.loc[(df['indrel'] == 99),'indrel'] = 0
    
    df.loc[(df['tiprel_1mes'] != 'A'),'tiprel_1mes'] = 'I'

    df.loc[(df['pais_residencia'] != 'ES'),'pais_residencia'] = 'O'           
    
    df.loc[(df['ind_empleado'] != 'N'),'ind_empleado'] = 'I'
    
    # Grouping Channels
    df['channel_grp1'] = df['canal_entrada']
    df.loc[-(df['canal_entrada'].isin(['KHE','KAT','KFC','KHK','KHQ','KFA','KHM','KHN','RED'])),'channel_grp1'] = df.loc[-(df['canal_entrada'].isin(['KHE','KAT','KFC','KHK','KHQ','KFA','KHM','KHN','RED'])),'channel_grp1'].str[:2]
    df.loc[-(df['channel_grp1'].isin(['KHE','KAT','KFC','KHK','KHQ','KFA','KHM','KHN','RED','KH','KA','KF','KC'])),'channel_grp1'] = df.loc[-(df['channel_grp1'].isin(['KHE','KAT','KFC','KHK','KHQ','KFA','KHM','KHN','RED','KH','KA','KF','KC'])),'channel_grp1'].str[:1]
    df.loc[(df['canal_entrada'].isnull()),'channel_grp1'] = '0'

    df.loc[df.nomprov.isnull(),'nomprov'] = 'NAN'

    df.loc[df['age'] > df['age'].quantile(.999),'age'] = df['age'].quantile(.999) #capping outliers       
    df['age_grp1'] = 'aA'
    df.loc[(df['age'] >= 20) & (df['age'] <= 25),'age_grp1'] = 'aB'
    df.loc[(df['age'] > 25) & (df['age'] <= 30),'age_grp1'] = 'aC'
    df.loc[(df['age'] > 30) & (df['age'] <= 35),'age_grp1'] = 'aD'
    df.loc[(df['age'] > 35) & (df['age'] <= 40),'age_grp1'] = 'aE'
    df.loc[(df['age'] > 40) & (df['age'] <= 45),'age_grp1'] = 'aF'           
    df.loc[(df['age'] > 45) & (df['age'] <= 50),'age_grp1'] = 'aG'
    df.loc[(df['age'] > 50) & (df['age'] <= 60),'age_grp1'] = 'aH'
    df.loc[(df['age'] > 60) & (df['age'] <= 75),'age_grp1'] = 'aI'    
    df.loc[(df['age'] > 75),'age_grp1'] = 'aJ'
    
    df.loc[df.sexo.isnull(),'sexo'] = 'V'
    
    df.loc[df.segmento.isnull(),'segmento'] = '02 - PARTICULARES'

    # Missing value imputation for 'renta'
    renta_by_prov_seg = pd.DataFrame(df.groupby(['nomprov','segmento'])['renta'].agg({'median'}))
    renta_by_prov_seg.reset_index(level=0, inplace=True)
    renta_by_prov_seg.reset_index(level=0, inplace=True)
    renta_by_prov_seg.columns = ['segmento','nomprov','med_renta1']
    
    df = pd.merge(df,renta_by_prov_seg,on = ['nomprov','segmento'],how = 'inner')
    
    renta_by_prov = pd.DataFrame(df.groupby(['segmento','sexo'])['renta'].quantile(.4))
    renta_by_prov.reset_index(level=0, inplace=True)
    renta_by_prov.reset_index(level=0, inplace=True)
    renta_by_prov.columns = ['sexo','segmento','med_renta2']
    
    df = pd.merge(df,renta_by_prov,on = ['segmento','sexo'],how = 'inner')
    
    df.loc[df.renta.isnull(),'renta'] = df.loc[df.renta.isnull(),'med_renta1']
    df.loc[df.renta.isnull(),'renta'] = df.loc[df.renta.isnull(),'med_renta2']
    df = df.drop(['med_renta1','med_renta2'],axis=1)
    df['income_grp1'] = pd.qcut(df['renta'], 6,labels=False)

    return(df)

df_train_new_0416 = data_prep_dmg(df_train_new_0416)
df_train_new_0615 = data_prep_dmg(df_train_new_0615)
df_sub = data_prep_dmg(df_sub)
                
# Addiing previous months transactions info (2 sets one is immediate last month and other last 3 months)
def add_prev_prod_purchase_info(df_train,df,last5_month):    
    for i in range(len(last5_month)):
        df_train_last5_prod = pd.DataFrame(df_train.ix[df_train.fecha_dato.isin([last5_month[i]]),:].groupby(['ncodpers'])[products].sum())
        df_train_last5_prod.columns = [str(col) + '_last' + str(i) for col in df_train_last5_prod.columns]
        df_train_last5_prod.reset_index(level=0, inplace=True)        
        df = pd.merge(df,df_train_last5_prod,on = ['ncodpers'],how = 'left')
        df = df.fillna(0)
    return(df)
    
df_train_new_0416 = add_prev_prod_purchase_info(df_train,df_train_new_0416,['2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28'])
df_train_new_0615 = add_prev_prod_purchase_info(df_train,df_train_new_0615,['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28'])
df_sub = add_prev_prod_purchase_info(df_train,df_sub,['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28'])

def add_last5_prod_purchase_info(df_train,df,last5_month):    
    df_train_tot_last5_prod = pd.DataFrame(df_train.ix[df_train.fecha_dato.isin(last5_month),:].groupby(['ncodpers'])[products].sum())
    df_train_tot_last5_prod.columns = [str(col) + '_total_last5' for col in df_train_tot_last5_prod.columns]
#    df_train_tot_last5_prod[df_train_tot_last5_prod>=1] = 1
    df_train_tot_last5_prod.reset_index(level=0, inplace=True)
    
    df = pd.merge(df,df_train_tot_last5_prod,on = ['ncodpers'],how = 'left')
    df = df.fillna(0)
    return(df)
    
df_train_new_0416 = add_last5_prod_purchase_info(df_train,df_train_new_0416,['2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28'])
df_train_new_0615 = add_last5_prod_purchase_info(df_train,df_train_new_0615,['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28'])
df_sub = add_last5_prod_purchase_info(df_train,df_sub,['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28'])

def add_last5_unique_prod_info(df_train,df,last5_month):    
    df_train_uni_last5_prod = pd.DataFrame(df_train.ix[df_train.fecha_dato.isin(last5_month),:].groupby(['ncodpers'])[products].sum())
    df_train_uni_last5_prod[df_train_uni_last5_prod>=1] = 1
    df_train_uni_last5_prod['unique_prod'] = df_train_uni_last5_prod.sum(axis=1)
    df_train_uni_last5_prod.reset_index(level=0, inplace=True) 
    df = pd.merge(df,df_train_uni_last5_prod[['ncodpers','unique_prod']],on = ['ncodpers'],how = 'left')
    df = df.fillna(0)
    return(df)
    
df_train_new_0416 = add_last5_unique_prod_info(df_train,df_train_new_0416,['2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28'])
df_train_new_0615 = add_last5_unique_prod_info(df_train,df_train_new_0615,['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28'])
df_sub = add_last5_unique_prod_info(df_train,df_sub,['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28'])

def add_monthly_new_prod(df_train,df,date_list):
    user_prod_monthly = df_train.ix[(df_train['fecha_dato'] == date_list[0]),products+['ncodpers']]
    user_prod_monthly = user_prod_monthly.fillna(0)
    for i in range(len(date_list)):
        if i == len(date_list)-1:
            break
        else:
            print i, date_list[i], i+1, date_list[i+1]
            temp = monthly_new_prod_users(df_train,date_list[i],date_list[i+1])
            temp[temp==1] = i+2
            temp = temp.fillna(0)
#            print temp.max()
            user_prod_monthly = user_prod_monthly.append(temp, ignore_index=True)
    user_prod_monthly = pd.DataFrame(user_prod_monthly.groupby(['ncodpers'])[products].max())
    user_prod_monthly.reset_index(level=0, inplace=True)
    user_prod_monthly['recent_activity_score'] = user_prod_monthly[products].max(axis=1)
    user_prod_monthly.columns = [str(col) + '_mnth' for col in user_prod_monthly.columns]            
    user_prod_monthly.rename(columns={'ncodpers_mnth': 'ncodpers'}, inplace=True)            
    df = pd.merge(df,user_prod_monthly,on = ['ncodpers'],how = 'left')
    df = df.fillna(0)
    return(df)

df_train_new_0416 = add_monthly_new_prod(df_train,df_train_new_0416,['2015-11-28','2015-12-28','2016-01-28','2016-02-28','2016-03-28'])
df_train_new_0615 = add_monthly_new_prod(df_train,df_train_new_0615,['2015-01-28','2015-02-28','2015-03-28','2015-04-28','2015-05-28'])
df_sub = add_monthly_new_prod(df_train,df_sub,['2016-01-28','2016-02-28','2016-03-28','2016-04-28','2016-05-28'])

df_train_new_0416['month'] = '0416'
df_train_new_0615['month'] = '0615'
df_train_new = df_train_new_0615.append(df_train_new_0416, ignore_index=True)

# Output from k_san_dp_15_v1_2.py
pred_train = pd.read_pickle(wdir + 'pred_train_1.pkl')
pred_sub = pd.read_pickle(wdir + 'pred_sub_1.pkl')

df_train_new = pd.merge(df_train_new,pred_train,on=['ncodpers','month'],how='inner')
df_sub = pd.merge(df_sub,pred_sub,on=['ncodpers'],how='inner')

#Input variables getting used in the model
cat_var = ['tenure_grp1','indresi','sexo','ind_empleado','tiprel_1mes','indext','channel_grp1','indfall','nomprov','segmento','age_grp1']
num_var = ['tenure','age','ind_actividad_cliente','indrel','income_grp1',
'ind_ahor_fin_ult1_last0', 'ind_aval_fin_ult1_last0','ind_cco_fin_ult1_last0', 'ind_cder_fin_ult1_last0',
'ind_cno_fin_ult1_last0', 'ind_ctju_fin_ult1_last0','ind_ctma_fin_ult1_last0', 'ind_ctop_fin_ult1_last0',
'ind_ctpp_fin_ult1_last0', 'ind_deco_fin_ult1_last0','ind_deme_fin_ult1_last0', 'ind_dela_fin_ult1_last0',
'ind_ecue_fin_ult1_last0', 'ind_fond_fin_ult1_last0','ind_hip_fin_ult1_last0', 'ind_plan_fin_ult1_last0',
'ind_pres_fin_ult1_last0', 'ind_reca_fin_ult1_last0','ind_tjcr_fin_ult1_last0', 'ind_valo_fin_ult1_last0',
'ind_viv_fin_ult1_last0', 'ind_nomina_ult1_last0','ind_nom_pens_ult1_last0', 'ind_recibo_ult1_last0',
'ind_ahor_fin_ult1_last1', 'ind_aval_fin_ult1_last1','ind_cco_fin_ult1_last1', 'ind_cder_fin_ult1_last1',
'ind_cno_fin_ult1_last1', 'ind_ctju_fin_ult1_last1','ind_ctma_fin_ult1_last1', 'ind_ctop_fin_ult1_last1',
'ind_ctpp_fin_ult1_last1', 'ind_deco_fin_ult1_last1','ind_deme_fin_ult1_last1', 'ind_dela_fin_ult1_last1',
'ind_ecue_fin_ult1_last1', 'ind_fond_fin_ult1_last1','ind_hip_fin_ult1_last1', 'ind_plan_fin_ult1_last1',
'ind_pres_fin_ult1_last1', 'ind_reca_fin_ult1_last1','ind_tjcr_fin_ult1_last1', 'ind_valo_fin_ult1_last1',
'ind_viv_fin_ult1_last1', 'ind_nomina_ult1_last1','ind_nom_pens_ult1_last1', 'ind_recibo_ult1_last1',
'ind_ahor_fin_ult1_last2', 'ind_aval_fin_ult1_last2','ind_cco_fin_ult1_last2', 'ind_cder_fin_ult1_last2',
'ind_cno_fin_ult1_last2', 'ind_ctju_fin_ult1_last2','ind_ctma_fin_ult1_last2', 'ind_ctop_fin_ult1_last2',
'ind_ctpp_fin_ult1_last2', 'ind_deco_fin_ult1_last2','ind_deme_fin_ult1_last2', 'ind_dela_fin_ult1_last2',
'ind_ecue_fin_ult1_last2', 'ind_fond_fin_ult1_last2','ind_hip_fin_ult1_last2', 'ind_plan_fin_ult1_last2',
'ind_pres_fin_ult1_last2', 'ind_reca_fin_ult1_last2','ind_tjcr_fin_ult1_last2', 'ind_valo_fin_ult1_last2',
'ind_viv_fin_ult1_last2', 'ind_nomina_ult1_last2','ind_nom_pens_ult1_last2', 'ind_recibo_ult1_last2',
'ind_ahor_fin_ult1_last3', 'ind_aval_fin_ult1_last3','ind_cco_fin_ult1_last3', 'ind_cder_fin_ult1_last3',
'ind_cno_fin_ult1_last3', 'ind_ctju_fin_ult1_last3','ind_ctma_fin_ult1_last3', 'ind_ctop_fin_ult1_last3',
'ind_ctpp_fin_ult1_last3', 'ind_deco_fin_ult1_last3','ind_deme_fin_ult1_last3', 'ind_dela_fin_ult1_last3',
'ind_ecue_fin_ult1_last3', 'ind_fond_fin_ult1_last3','ind_hip_fin_ult1_last3', 'ind_plan_fin_ult1_last3',
'ind_pres_fin_ult1_last3', 'ind_reca_fin_ult1_last3','ind_tjcr_fin_ult1_last3', 'ind_valo_fin_ult1_last3',
'ind_viv_fin_ult1_last3', 'ind_nomina_ult1_last3','ind_nom_pens_ult1_last3', 'ind_recibo_ult1_last3',
'ind_ahor_fin_ult1_last4', 'ind_aval_fin_ult1_last4','ind_cco_fin_ult1_last4', 'ind_cder_fin_ult1_last4',
'ind_cno_fin_ult1_last4', 'ind_ctju_fin_ult1_last4','ind_ctma_fin_ult1_last4', 'ind_ctop_fin_ult1_last4',
'ind_ctpp_fin_ult1_last4', 'ind_deco_fin_ult1_last4','ind_deme_fin_ult1_last4', 'ind_dela_fin_ult1_last4',
'ind_ecue_fin_ult1_last4', 'ind_fond_fin_ult1_last4','ind_hip_fin_ult1_last4', 'ind_plan_fin_ult1_last4',
'ind_pres_fin_ult1_last4', 'ind_reca_fin_ult1_last4','ind_tjcr_fin_ult1_last4', 'ind_valo_fin_ult1_last4',
'ind_viv_fin_ult1_last4', 'ind_nomina_ult1_last4','ind_nom_pens_ult1_last4', 'ind_recibo_ult1_last4',
'ind_ahor_fin_ult1_total_last5', 'ind_aval_fin_ult1_total_last5','ind_cco_fin_ult1_total_last5', 'ind_cder_fin_ult1_total_last5',
'ind_cno_fin_ult1_total_last5', 'ind_ctju_fin_ult1_total_last5','ind_ctma_fin_ult1_total_last5', 'ind_ctop_fin_ult1_total_last5',
'ind_ctpp_fin_ult1_total_last5', 'ind_deco_fin_ult1_total_last5','ind_deme_fin_ult1_total_last5', 'ind_dela_fin_ult1_total_last5',
'ind_ecue_fin_ult1_total_last5', 'ind_fond_fin_ult1_total_last5','ind_hip_fin_ult1_total_last5', 'ind_plan_fin_ult1_total_last5',
'ind_pres_fin_ult1_total_last5', 'ind_reca_fin_ult1_total_last5','ind_tjcr_fin_ult1_total_last5', 'ind_valo_fin_ult1_total_last5',
'ind_viv_fin_ult1_total_last5', 'ind_nomina_ult1_total_last5','ind_nom_pens_ult1_total_last5', 'ind_recibo_ult1_total_last5',
'ind_ahor_fin_ult1_mnth', 'ind_aval_fin_ult1_mnth','ind_cco_fin_ult1_mnth', 'ind_cder_fin_ult1_mnth',
'ind_cno_fin_ult1_mnth', 'ind_ctju_fin_ult1_mnth','ind_ctma_fin_ult1_mnth', 'ind_ctop_fin_ult1_mnth',
'ind_ctpp_fin_ult1_mnth', 'ind_deco_fin_ult1_mnth','ind_deme_fin_ult1_mnth', 'ind_dela_fin_ult1_mnth',
'ind_ecue_fin_ult1_mnth', 'ind_fond_fin_ult1_mnth','ind_hip_fin_ult1_mnth', 'ind_plan_fin_ult1_mnth',
'ind_pres_fin_ult1_mnth', 'ind_reca_fin_ult1_mnth','ind_tjcr_fin_ult1_mnth', 'ind_valo_fin_ult1_mnth',
'ind_viv_fin_ult1_mnth', 'ind_nomina_ult1_mnth','ind_nom_pens_ult1_mnth', 'ind_recibo_ult1_mnth',
'recent_activity_score_mnth','unique_prod', "['ind_cco_fin_ult1', 'ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_cco_fin_ult1', 'ind_cno_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_ctma_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_cco_fin_ult1', 'ind_ctma_fin_ult1', 'ind_recibo_ult1']","['ind_cco_fin_ult1', 'ind_ctma_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_ctma_fin_ult1']","['ind_cco_fin_ult1', 'ind_ctop_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_ctpp_fin_ult1']","['ind_cco_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_dela_fin_ult1']","['ind_cco_fin_ult1', 'ind_ecue_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_cco_fin_ult1', 'ind_ecue_fin_ult1', 'ind_recibo_ult1']","['ind_cco_fin_ult1', 'ind_ecue_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_fond_fin_ult1']","['ind_cco_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1']","['ind_cco_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_tjcr_fin_ult1']","['ind_cco_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_cco_fin_ult1', 'ind_nom_pens_ult1', 'ind_reca_fin_ult1']","['ind_cco_fin_ult1', 'ind_nom_pens_ult1']",
 "['ind_cco_fin_ult1', 'ind_reca_fin_ult1', 'ind_recibo_ult1']","['ind_cco_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_reca_fin_ult1']","['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_valo_fin_ult1']","['ind_cco_fin_ult1', 'ind_recibo_ult1']",
 "['ind_cco_fin_ult1', 'ind_tjcr_fin_ult1']","['ind_cco_fin_ult1', 'ind_valo_fin_ult1']",
 "['ind_cco_fin_ult1']","['ind_cder_fin_ult1']","['ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_recibo_ult1']","['ind_cno_fin_ult1', 'ind_ctma_fin_ult1']",
 "['ind_cno_fin_ult1', 'ind_ctop_fin_ult1']","['ind_cno_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_nom_pens_ult1']",
 "['ind_cno_fin_ult1', 'ind_ctpp_fin_ult1']","['ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 'ind_recibo_ult1']","['ind_cno_fin_ult1', 'ind_ecue_fin_ult1']",
 "['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1', 'ind_recibo_ult1']",
 "['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1']",
 "['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1']",
 "['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_tjcr_fin_ult1']","['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']",
 "['ind_cno_fin_ult1', 'ind_nom_pens_ult1']","['ind_cno_fin_ult1', 'ind_reca_fin_ult1']","['ind_cno_fin_ult1', 'ind_recibo_ult1', 'ind_tjcr_fin_ult1']","['ind_cno_fin_ult1', 'ind_recibo_ult1']","['ind_cno_fin_ult1', 'ind_tjcr_fin_ult1']","['ind_cno_fin_ult1']",
 "['ind_ctju_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_ctju_fin_ult1']",
 "['ind_ctma_fin_ult1', 'ind_dela_fin_ult1']","['ind_ctma_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1']",
 "['ind_ctma_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_ctma_fin_ult1', 'ind_recibo_ult1']",
 "['ind_ctma_fin_ult1', 'ind_tjcr_fin_ult1']","['ind_ctma_fin_ult1', 'ind_valo_fin_ult1']","['ind_ctma_fin_ult1']",
 "['ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1']","['ind_ctop_fin_ult1', 'ind_dela_fin_ult1']","['ind_ctop_fin_ult1', 'ind_ecue_fin_ult1']","['ind_ctop_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_ctop_fin_ult1', 'ind_nom_pens_ult1']",
 "['ind_ctop_fin_ult1', 'ind_reca_fin_ult1']","['ind_ctop_fin_ult1', 'ind_recibo_ult1']","['ind_ctop_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_ctop_fin_ult1']","['ind_ctpp_fin_ult1', 'ind_ecue_fin_ult1']","['ind_ctpp_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_ctpp_fin_ult1', 'ind_reca_fin_ult1']","['ind_ctpp_fin_ult1', 'ind_recibo_ult1']","['ind_ctpp_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_ctpp_fin_ult1']","['ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1']",
 "['ind_dela_fin_ult1', 'ind_ecue_fin_ult1']","['ind_dela_fin_ult1', 'ind_fond_fin_ult1', 'ind_recibo_ult1']",
 "['ind_dela_fin_ult1', 'ind_fond_fin_ult1']","['ind_dela_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_dela_fin_ult1', 'ind_nom_pens_ult1']","['ind_dela_fin_ult1', 'ind_reca_fin_ult1']",
 "['ind_dela_fin_ult1', 'ind_recibo_ult1']","['ind_dela_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_dela_fin_ult1']","['ind_ecue_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1']",
 "['ind_ecue_fin_ult1', 'ind_nom_pens_ult1', 'ind_nomina_ult1']","['ind_ecue_fin_ult1', 'ind_nom_pens_ult1']",
 "['ind_ecue_fin_ult1', 'ind_reca_fin_ult1']","['ind_ecue_fin_ult1', 'ind_recibo_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_ecue_fin_ult1', 'ind_recibo_ult1']","['ind_ecue_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_ecue_fin_ult1', 'ind_valo_fin_ult1']","['ind_ecue_fin_ult1']",
 "['ind_fond_fin_ult1', 'ind_nom_pens_ult1']","['ind_fond_fin_ult1', 'ind_reca_fin_ult1']",
 "['ind_fond_fin_ult1', 'ind_recibo_ult1']","['ind_fond_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_fond_fin_ult1', 'ind_valo_fin_ult1']","['ind_fond_fin_ult1']","['ind_hip_fin_ult1']",
 "['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1', 'ind_recibo_ult1']",
 "['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_reca_fin_ult1']",
 "['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1', 'ind_tjcr_fin_ult1']","['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_recibo_ult1']","['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_nom_pens_ult1', 'ind_nomina_ult1', 'ind_valo_fin_ult1']","['ind_nom_pens_ult1', 'ind_nomina_ult1']",
 "['ind_nom_pens_ult1', 'ind_reca_fin_ult1', 'ind_recibo_ult1']","['ind_nom_pens_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_nom_pens_ult1', 'ind_reca_fin_ult1']","['ind_nom_pens_ult1', 'ind_recibo_ult1']","['ind_nom_pens_ult1', 'ind_tjcr_fin_ult1']",
 "['ind_nom_pens_ult1', 'ind_valo_fin_ult1']","['ind_nom_pens_ult1']","['ind_nomina_ult1']","['ind_plan_fin_ult1']","['ind_pres_fin_ult1']","['ind_reca_fin_ult1', 'ind_recibo_ult1', 'ind_tjcr_fin_ult1']","['ind_reca_fin_ult1', 'ind_recibo_ult1']",
 "['ind_reca_fin_ult1', 'ind_tjcr_fin_ult1']","['ind_reca_fin_ult1']",
 "['ind_recibo_ult1', 'ind_tjcr_fin_ult1']","['ind_recibo_ult1', 'ind_valo_fin_ult1']","['ind_recibo_ult1']",
 "['ind_tjcr_fin_ult1', 'ind_valo_fin_ult1']","['ind_tjcr_fin_ult1']","['ind_valo_fin_ult1']","['ind_viv_fin_ult1']"]

#num_var = [col for col in num_var if 'deco' not in col]
#num_var = [col for col in num_var if 'aval' not in col]
#num_var = [col for col in num_var if 'deme' not in col]
#num_var = [col for col in num_var if 'ahor' not in col]

drop_targets = ['ind_deco_fin_ult1','ind_deme_fin_ult1','ind_aval_fin_ult1','ind_ahor_fin_ult1']
df_train_new = df_train_new.drop(drop_targets,axis=1)
df_train_new = df_train_new.ix[df_train_new[[x for x in products if x not in drop_targets]].sum(axis=1)!=0,:]
                               
# Creating dummy binary variables for categorical variable and converting everything to numpy matrix
df_train_new_arr = one_hot_encod(df_train_new,cat_var,num_var)
df_train_new_arr_normed = df_train_new_arr / df_train_new_arr.max(axis=0)
df_sub_arr = one_hot_encod(df_sub,cat_var,num_var)
df_sub_arr_normed = df_sub_arr / df_sub_arr.max(axis=0)
target = df_train_new[[x for x in products if x not in drop_targets]].as_matrix()

rms = rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Model Building 
def keras_model():
    # create model
    model = Sequential()
    model.add(Dense(len(df_train_new_arr[0]*2), input_dim=len(df_train_new_arr[0]), init='uniform', activation='tanh'))
    model.add(BatchNormalization())#batch normalization reduces the dependence on right initialization of weights, removes white noise, can totally elimiate the need for drop out
    model.add(Dropout(0.5))
    model.add(Dense(len(df_train_new_arr[0])*3, init='uniform', activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(len(df_train_new_arr[0])*1, init='uniform', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(target[0]), init='uniform', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=keras_model, nb_epoch=49, batch_size=500, verbose=1)
np.random.seed(123)
estimator.fit(df_train_new_arr_normed, target,verbose=1,validation_split=0.3,show_accuracy=True)
predictions = estimator.predict_proba(df_sub_arr_normed)
pred = pd.DataFrame(data=predictions,columns=[x for x in products if x not in drop_targets])
pred['ncodpers'] = df_sub['ncodpers']

# Removing items already present in May-16
pred_T = pd.melt(pred, id_vars="ncodpers",var_name="Var",value_name="Val")

df_train_0516 = pd.DataFrame(df_train.ix[df_train.fecha_dato=='2016-05-28',:].groupby(['ncodpers'])[products].sum())
df_train_0516.reset_index(level=0, inplace=True)
df_train_0516_T = pd.melt(df_train_0516, id_vars="ncodpers",var_name="Var",value_name="Val1")
df_train_0516_T = df_train_0516_T.ix[df_train_0516_T.Val1 > 0,:]

pred_not0516 = pd.merge(pred_T,df_train_0516_T,on=['ncodpers','Var'],how='left')
pred_not0516 = pred_not0516.ix[pred_not0516.Val1.isnull(),:]
pred_not0516 = pred_not0516.sort(['ncodpers','Val'], ascending=[True,False])
pred_not0516_top7 = pred_not0516.groupby('ncodpers').head(7)
user_row_number = pred_not0516_top7.groupby('ncodpers')
pred_not0516_top7['Name'] = user_row_number['Val'].cumcount()
pred_not0516_top7_T = pred_not0516_top7.pivot(index='ncodpers',columns='Name',values='Var')
pred_not0516_top7_T.reset_index(level=0, inplace=True)
pred_not0516_top7_T['added_products'] = pred_not0516_top7_T[0].map(str) + " " + pred_not0516_top7_T[1].map(str) + " " + pred_not0516_top7_T[2].map(str)+ " " + pred_not0516_top7_T[3].map(str)+ " " + pred_not0516_top7_T[4].map(str)+ " " + pred_not0516_top7_T[5].map(str)+ " " + pred_not0516_top7_T[6].map(str)
# Final Submission 
sub = pred_not0516_top7_T[['ncodpers','added_products']]
sub.to_csv(wdir + 'sub.csv', index=False)