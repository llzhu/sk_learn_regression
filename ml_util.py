import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from io import StringIO, BytesIO
import os
import shutil
import re
from dataclasses import dataclass, field
from typing import List
import math
from icecream import ic
import base64
from itertools import chain
import boto3
import pickle
from typing import Tuple


s3client = boto3.client(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)

s3resource = boto3.resource(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)

def pickle_to_s3(data_obj, bucket, key):
    # Serialize to memory
    buffer = BytesIO()
    pickle.dump(data_obj, buffer)
    buffer.seek(0)

    # Upload to S3
    s3client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )

def get_from_s3(bucket, key):
    # Download pickle from S3
    pickle_obj = s3client.get_object(Bucket=bucket, Key=key)

    # Deserialize
    buffer = BytesIO(pickle_obj["Body"].read())
    data_obj = pickle.load(buffer)

    return data_obj

def any_contents(bucket, prefix):
    response = s3client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/"
    )

    ic(response.get("Contents", []))
    return len(response.get("Contents", []))>0
    
    

def get_df_from_s3csv(bucket, key):
    # Get object from S3
    obj = s3client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(BytesIO(obj["Body"].read()))
    return df



# algorithms
MODEL_NN = 'Neural_Network'
MODEL_LBR = 'Linear_Bayesian_Ridge'
MODEL_RF = 'Random_Forest'
MODEL_HGB = 'Hist_Gradient_Boost'

# discriptors
FP_ONLY = 'Morgan_FP'
ADD_RDKIT_DESCRIPTORS = 'Morgan_FP_2D_Descriptors'
RDKIT_DESCRIPTORS_ONLY = '2D_Descriptors'

# Studies/dataset
DELANEY = 'Solubility_Delaney'
THROBIN_IC50 = 'Thrombin_IC50'
AD_HOC = 'ad_hoc'

RADIUS = 3 
FP_SIZE = 4096

SMI_LIST = 'SMILES lists'
FILE_UPLOAD = 'File Upload'

SMILES = 'SMILES'
DO_NOT_HIGHLIGHT = "Do not highlight"
HIGHLIGHT_ALL = "Highlight All"
HIGHLIGHT_UNIQUE = "Highlight Unique"
COMPOUND_ID = 'Compound_ID'
STRUCTURE = 'Compound'
CHEMBL_UNIT = 'standard_units'
CHEMBL_SMILES = 'canonical_smiles'
CHEMBL_CMPD_ID = 'molecule_chembl_id'

MY_MODEL = 'My model'
MASTER_MODEL = 'Master Model'
MODEL_OPTIONS = [MY_MODEL, MASTER_MODEL]



@dataclass
class AppVars:
    study: str
    dataset_shape: Tuple[int,int] = (1, 1)
    orig_col_name: str = ''  # column name of the original data
    expt_col_name: str = ''  # column name used in training/predication. = or log of orig_col_name
    apply_log: bool = True   # The data are log scaled for training; expt_col_name=log(orig_col_name)
     

    
@dataclass
class ModelDesc:
    X_desc: str = ''
    X_cols: List[str] = field(default_factory=list)
    X_scaler: StandardScaler = None
    y_scaler: StandardScaler = None
    class_name: str = ''
    model: object = None
    

@dataclass
class ModelData:
    X: pd.DataFrame
    y: List[float]

@dataclass
class Env:
    src_data: str
    app_data: str
    admins: List[str] = field(default_factory=list)
    modelers: List[str] = field(default_factory=list)
    s3_bucket: str = ''



def get_prefix(env:Env, app_vars:AppVars, model_desc:ModelDesc):
    return f'{env.app_data}/{app_vars.study}/{model_desc.class_name}/{model_desc.X_desc}/'


def delete_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_df_csv(df):
    f = StringIO()
    df.to_csv(f, index=False)
    return f

def convert_ugperml_to_um(row):
    if row['unit'] == 'ug/mL':
        return (row['value']/row['mw'])*1000.0
    else:
        return row['value']

def convert_df_csv(df, index=False):
    return df.to_csv(index=index).encode('utf-8')




# def get_chemdl_activity_df(standard_type, target_organism):
#
#     activity = new_client.activity
#     activity = activity.filter(
#             standard_type=standard_type,
#             target_organism=target_organism,
#             dard_value__isnull=False,
#             standard_value__gt=0,
#             standard_units='mL.min-1.kg-1').only(
#             "molecule_chembl_id",
#             "canonical_smiles",
#             "target_organism",
#             "standard_type",
#             "standard_relation",
#             "standard_value",
#             "standard_units")
#     df = pd.DataFrame(activity)
#     df['mw'] = df['canonical_smiles'].apply(lambda x: ExactMolWt(Chem.MolFromSmiles(x)))
#
#     return df

@st.cache_data
def standarize(df_input: pd.DataFrame, study:str, value_column:str, apply_log:bool)-> tuple[pd.DataFrame, str]:
    
    df_output = None
    if study == THROBIN_IC50:
        df_output = df_input[df_input[CHEMBL_UNIT]=='nM']
        df_output = df_output[~df_output[CHEMBL_SMILES].str.contains('.', regex=False, na=False)]
        df_output[value_column] = df_output[value_column].astype(float)
        if apply_log:
            expt_col_name = 'log_IC50'
            df_output[expt_col_name] = df_output[value_column].apply(math.log10)
        else:
            expt_col_name = 'IC50'
            df_output[expt_col_name] = df_output[value_column]

        column_map = {
                CHEMBL_CMPD_ID: COMPOUND_ID,
                CHEMBL_SMILES: SMILES
            }
        df_output = df_output.rename(columns=column_map) 
        col_output = [COMPOUND_ID, expt_col_name, SMILES]
        df_output = df_output[col_output] 
        df_output[expt_col_name] = df_output[expt_col_name].astype(float)


    return df_output, expt_col_name




def get_list(inputs: str)->list[str]: 
    input_list = []
    if inputs:
        input_list = re.split(',|\n', inputs)
        input_list = [input for input in input_list if input.strip()]
    return input_list

def get_floor(in_num: float, floor: float)-> float: 
    out_num = in_num
    if in_num < floor:
        out_num = floor
    return out_num



# class MorganFP:
#     def __init__(self, radius, fp_size):
#         self.radius = radius
#         self.fp_size = fp_size
#         self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
#             radius=self.radius,
#             fpSize=self.fp_size
#         )

#     def __call__(self, mol):
#         return np.array(self.morgan_gen.GetFingerprint(mol))



# def get_rdkit_fp(morgan_fp:MorganFP, mol_list, radius, fp_size):
#     X = [morgan_fp(mol) for mol in mol_list]
#     X = pd.DataFrame(data=X)  # Make it a dataframe
#     return X

def remove_low_variance(input_data, threshold=0.1) -> pd.DataFrame:
    # input_data expacted to be np.ndarray or pd.Dataframe
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(data=input_data)  
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

def get_rdkit_fp(morgan_gen, mol_list):
    X = [ np.array(morgan_gen.GetFingerprint(mol)) for mol in mol_list]
    X = pd.DataFrame(data=X)  # Make it a dataframe
    return X

def get_rdkit_descriptors(mol_list):
    descriptor_names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol_descriptors = []
    for mol in mol_list:
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)

    df = pd.DataFrame(mol_descriptors, columns=descriptor_names)
    return df
    

def get_all_descriptors(mol_list, radius, fp_size, descriptor_sel, reduced=True):

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    if descriptor_sel == FP_ONLY:
        X_FP = get_rdkit_fp(morgan_gen, mol_list)
        if reduced:
            X_FP = remove_low_variance(X_FP)
        X = X_FP


    if descriptor_sel == ADD_RDKIT_DESCRIPTORS:
        X_FP = get_rdkit_fp(morgan_gen, mol_list)
        if reduced:
            X_FP = remove_low_variance(X_FP)

        X_DESC_2D = get_rdkit_descriptors(mol_list)
        if reduced:
            X_DESC_2D = remove_low_variance(X_DESC_2D)

        X = pd.concat([X_FP, X_DESC_2D], axis=1, join='inner')


    if descriptor_sel == RDKIT_DESCRIPTORS_ONLY:
        X_DESC_2D = get_rdkit_descriptors(mol_list)
        if reduced:
            X_DESC_2D = remove_low_variance(X_DESC_2D)
        X = X_DESC_2D

    X.columns = X.columns.astype(str) 
    return X


@st.cache_data
def moltosvg(mol, molSize = (800,400), kekulize = False, highlight_sub=None, highlight_mode=DO_NOT_HIGHLIGHT):
    
    if  highlight_sub == None: # Cannot highlight if highlight_sub not provided
        highlight_mode=DO_NOT_HIGHLIGHT
    
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    
    if highlight_mode == DO_NOT_HIGHLIGHT:
        drawer.DrawMolecule(mc)
    elif highlight_mode in (HIGHLIGHT_UNIQUE, HIGHLIGHT_ALL):
        highlight_tt = mc.GetSubstructMatches(highlight_sub)
        hightlight_shape = np.shape(highlight_tt)
        if hightlight_shape[0] == 1:
            highlight_tuple = tuple(chain.from_iterable(highlight_tt))
            drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
        else:
            if highlight_mode == HIGHLIGHT_UNIQUE:
                drawer.DrawMolecule(mc)
            elif highlight_mode == HIGHLIGHT_ALL:
                highlight_tuple = tuple(chain.from_iterable(highlight_tt))
                drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = rf'<img src="data:image/svg+xml;base64, {b64}"/>'
    return html