import streamlit as st
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from ml_util import *
from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing

start = timer()

if 'new_model' in st.session_state:
    new_model = st.session_state['new_model']
else:
    st.write('Please go back to home page to set up a model to create.')
    st.stop()


if not new_model:
    st.write(f"An existing model will be used.")
    st.stop()

if 'env' in st.session_state:
    env:Env = st.session_state['env']

if 'app_vars' in st.session_state:
    app_vars:AppVars = st.session_state['app_vars']

if 'model_desc' in st.session_state:
    model_desc:ModelDesc = st.session_state['model_desc']

if 'model_data' in st.session_state:
    model_data:ModelData = st.session_state['model_data']


class_name = model_desc.class_name

st.write(f"Overall Prediction Accuracy of {class_name} on {app_vars.study} using {model_desc.X_desc} as features:")
summery_empty = st.empty()
summery_empty.progress(0.01)
summary_container = st.container()


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
epochs = 100

n_test = int(100/k)
n_train = 100 - n_test
st.write(f'Detailed R2 scores and Root mean square error for different train({n_train})/test({n_test}) selections')

reg_index = 0
r2_test_list = []
r2_train_list = []
rmse_test_list = []
rmse_train_list = []

X = model_data.X
y = model_data.y
y_scaler = model_desc.y_scaler
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    y_scaler = preprocessing.StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    X_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    

    if class_name in [MODEL_L3, MODEL_L4]:
        
        model = model_factory(class_name, input_dim=len(model_desc.X_cols))
        model.to(DEVICE)
        torch_train_batch(model, 
                          criterion = nn.MSELoss(), 
                          optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                          epochs=epochs, 
                          dataset=TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), 
                          batch_size=32)
        # torch_train(model, 100, torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        model.eval()
        with torch.no_grad():
            y_train_pred = model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
            y_test_pred = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).detach().cpu().numpy()
    else:
        if class_name ==MODEL_NN:
            model = model_factory(class_name, hidden_layer_sizes=(256, 128), max_iter=epochs)
        else:
            model = model_factory(class_name)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_pred = y_train_pred.reshape(-1,1)
        y_test_pred = model.predict(X_test)
        y_test_pred = y_test_pred.reshape(-1,1)

    y_train = y_scaler.inverse_transform(y_train)
    y_train_pred = y_scaler.inverse_transform(y_train_pred)

    y_test = y_scaler.inverse_transform(y_test)
    y_test_pred = y_scaler.inverse_transform(y_test_pred)


    st.write('***')
    c1, c2 = st.columns(2)

    with c1:
        r2_train = r2_score(y_train, y_train_pred)
        r2_train_list.append(r2_train)
        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_train_list.append(rmse_train)

        st.write(f'R2: {round(r2_train,2)}; RSME: {round(rmse_train,2)}')

        plt.figure(reg_index)
        y_train_flat = y_train.reshape(-1)
        y_train_pred_flat = y_train_pred.reshape(-1)
        plt.scatter(x=y_train_flat, y=y_train_pred_flat, c="#7CAE00", alpha=0.3)

        z = np.polyfit(y_train_pred_flat, y_train_pred_flat, 1)
        p = np.poly1d(z)
        plt.plot(y_train_pred_flat, p(y_train_pred_flat), "#F8766D")

        plt.ylabel(f'Predicted {app_vars.expt_col_name}')
        plt.xlabel(f'Experimental {app_vars.expt_col_name}')

        st.pyplot(plt)

    with c2:
        r2_test = r2_score(y_test, y_test_pred)
        r2_test_list.append(r2_test)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)
        rmse_test_list.append(rmse_test)

        st.write(f'R2: {round(r2_test,2)}; RSME: {round(rmse_test,2)}')

        plt.figure(reg_index+1)
        y_test_flat = y_test.reshape(-1)
        y_test_pred_flat = y_test_pred.reshape(-1)
        plt.scatter(x=y_test_flat, y=y_test_pred_flat, c="#7CAE00", alpha=0.3)

        z_test = np.polyfit(y_test_flat, y_test_pred_flat, 1)
        p_test = np.poly1d(z_test)
        plt.plot(y_test_pred_flat, p_test(y_test_pred_flat), "#F8766D")

        plt.ylabel(f'Predicted {app_vars.expt_col_name}')
        plt.xlabel(f'Experimental {app_vars.expt_col_name}')

        st.pyplot(plt)

    reg_index += 2
    summery_empty.progress(reg_index/(k * 2))

end = timer()

# Train with the whole data set

y_scaler = preprocessing.StandardScaler().fit(y)
y = y_scaler.transform(y)
    
X_scaler = preprocessing.StandardScaler().fit(X)
X = X_scaler.transform(X)
    
if class_name in [MODEL_L3, MODEL_L4]:
    model = model_factory(class_name, input_dim=len(model_desc.X_cols))
    model.to(DEVICE)
    # torch_train(model, 100, torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    torch_train_batch(model, 
                        criterion = nn.MSELoss(), 
                        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                        epochs=epochs, 
                        dataset=TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), 
                        batch_size=32)
    model.to('cpu')
else:
    if class_name ==MODEL_NN:
        model = model_factory(class_name, hidden_layer_sizes=(256, 128), max_iter=epochs)
    else:
        model = model_factory(class_name)

    model.fit(X, y)

k_fold_rmse = round(np.mean(rmse_test_list),2)
k_fold_rmse_train = round(np.mean(rmse_test_list),2)
k_fold_r2 = round(np.mean(r2_test_list),2)
k_fold_r2_train = round(np.mean(r2_train_list),2)

model_desc.model = model   # populate model_desc with trained model
model_desc.X_scaler = X_scaler
model_desc.y_scaler = y_scaler
model_desc.k_fold_rmse = k_fold_rmse
model_desc.k_fold_r2 = k_fold_r2

key_prefix = f'{env.app_data}/{app_vars.study}/{class_name}/{model_desc.X_desc}'
key_model_desc = f'{key_prefix}/model_desc.pkl'
key_model_data = f'{key_prefix}/model_data.pkl'
pickle_to_s3(model_desc, env.s3_bucket, key_model_desc)
pickle_to_s3(model_data, env.s3_bucket, key_model_data)

with summary_container:
    st.write(f'Elapsed time for the {k} trainings: {timedelta(seconds=end - start)}')
    col1, col2 = st.columns(2)
    col1.write(f"Overall R2 value for train set: {k_fold_r2_train}")
    col1.write(f"Average RMSE for train set: {k_fold_rmse_train}")

    col2.write(f"Overall R2 value for test set: {k_fold_r2}")
    col2.write(f"Average RMSE for test set: {k_fold_rmse}")

    st.write(f'''A {class_name} model on {app_vars.study} has been also trained with all dataset.\n
         The trained model with meta data have been writen to {env.s3_bucket} s3 bucket:\n
         {key_prefix}
         ''')

