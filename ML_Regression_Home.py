import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
from rdkit import Chem
from sklearn import preprocessing
from ml_util import *
from ml_comp import *


st.set_page_config(page_title='SciKit-Learn Regression', layout='wide')

env = Env(  st.secrets['src_data'],
            st.secrets['app_data'],
            st.secrets['admins'],       
            st.secrets['modelers'],
            st.secrets['s3_bucket'],
        ) 


app_header()

study, apply_log, X_desc, algorithm, excluded_list, new_model = app_setup() 

model = None
class_name = ''
if algorithm == MODEL_HGB:
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    class_name = 'HistGradientBoostingRegressor'
elif algorithm == MODEL_LBR:
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge()
    class_name = 'BayesianRidge'
elif algorithm == MODEL_NN:
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(1000), activation='relu', max_iter=1000, shuffle=True)
    class_name = 'MLPRegressor'
elif algorithm == MODEL_RF:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    class_name = 'RandomForestRegressor'
elif algorithm == MODEL_TORCH:
    # it is a special case
    class_name = MODEL_TORCH

st.session_state['new_model'] = new_model
st.session_state['env'] = env

if study == '--':
    st.error('You must select a dataset to create/upload a model.' )
    st.stop()
       
if not new_model:
    # These basic data are still need to properly load the existing models
    app_vars = AppVars(study=study, apply_log=apply_log)
    model_desc = ModelDesc(X_desc=X_desc, class_name=class_name, model=model)
    st.session_state['app_vars'] = app_vars   
    st.session_state['model_desc'] = model_desc

    st.write(f"An existing model for {app_vars.study} will be used.")
    st.stop()

warning_container = st.container()

df_g = None
bin_0 = [0.1]
if study == DELANEY:
    # df_g = os.path.join(env.src_data, 'delaney.csv')
    df_g =  get_df_from_s3csv(env.s3_bucket, f'{env.src_data}/delaney.csv')
    df_g = df_g[['Compound ID', 'log_M', 'SMILES']]
    expt_col_name = 'log_Solubility_M'
    orig_col_name = 'Solubility_M'
    df_g = df_g.rename(columns={'log_M':expt_col_name})

    apply_log = True   # Special logic - we still want to see the non-log data
    
elif study == THROBIN_IC50:
    # activity = new_client.activity
    # data_ic50 = (activity.filter(target_chembl_id="CHEMBL204").filter(standard_type='IC50')
    #                     .filter(standard_relation='=')
    #                     .only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units'])
    #             )
    # data_ic50 = get_df_from_s3csv(bucket, key)

    # df_g = pd.DataFrame(data_ic50)

    df_g = df_g =  get_df_from_s3csv(env.s3_bucket, f'{env.src_data}/thrombin_ic50.csv')
    orig_col_name = 'IC50'
    df_g = df_g.rename(columns={'standard_value': orig_col_name})

    (df_g, expt_col_name)  = standarize(df_g, THROBIN_IC50, orig_col_name, apply_log)
    
    if not apply_log:
        bin_0 = 100
   
elif study == AD_HOC:
    df_g, expt_col_name = side_data_file_upload(warning_container=warning_container)
    

if df_g is not None:
    csv = convert_df_csv(df_g)
    st.sidebar.download_button("Download Smiles file", data=csv, file_name=f'smiles_{expt_col_name}.csv', mime='text/csv')


chem_list = [Chem.MolFromSmiles(smiles) for smiles in df_g.SMILES]
X = get_all_descriptors(chem_list, radius=RADIUS, fp_size=FP_SIZE, descriptor_sel=X_desc, reduced=True)
X_cols = X.columns




if algorithm == MODEL_TORCH:
    model = L3Model(len(X_cols), 256, 128)

y = df_g[expt_col_name].values.reshape(-1, 1)
y_scaler = preprocessing.StandardScaler().fit(y)
y = y_scaler.transform(y)

X_scaler = preprocessing.StandardScaler().fit(X)
X = X_scaler.transform(X)
X = pd.DataFrame(data=X)

# model.fit(X, y)

app_vars = AppVars(study, X.shape, orig_col_name, expt_col_name, apply_log)
model_desc = ModelDesc(X_desc, X_cols, X_scaler, y_scaler, class_name, model)
model_data = ModelData(X, y)


# key_prefix = f'{env.app_data}/{study}/{algorithm}/{X_desc}'
# key_model_desc = f'{key_prefix}/model_desc.pkl'
# key_model_data = f'{key_prefix}/model_data.pkl'
# pickle_to_s3(model_desc, env.s3_bucket, key_model_desc)
# pickle_to_s3(model_data, env.s3_bucket, key_model_data)


# # save them to session state

st.session_state['app_vars'] = app_vars   
st.session_state['model_desc'] = model_desc
st.session_state['model_data'] = model_data






c_data, c_fig = st.columns(2)
with c_data:
    st.write('All Feature Shapes = ', X.shape)
    st.dataframe(df_g)

with c_fig:
    y_0 = [df_g[expt_col_name]]
    t_0 = [expt_col_name]
    st.write(f'{expt_col_name} Distribution')
    fig_1 = ff.create_distplot(y_0, t_0, bin_size=bin_0)
    st.plotly_chart(fig_1)
