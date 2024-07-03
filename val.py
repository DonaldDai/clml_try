import argparse
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import contextlib

from sklearn.metrics import roc_auc_score, mean_squared_error

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, PandasTools
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display, HTML

from shutil import copyfile

import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task, Dataset, StorageManager
import argparse
import pickle

# set plotting parameters
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

def smiles_to_mols(query_smiles):
    mols = []
    # ipdb.set_trace()

        # Open a file for writing the output
    with open("smile_ouput.txt", 'w') as file:
    #     # Redirect standard output to the file
    #     original_stdout = sys.stdout
    #     sys.stdout = file
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):

            for smile in query_smiles:
                try: 
                    print("processing smiles: ", smile)
                    mols.append(Chem.MolFromSmiles(smile))
                except Exception as e:
                    print("Invalid SMILES: ", smile)
                    print("Error:", str(e))
                    file.flush()  

            # # Reset standard output to original
            # sys.stdout = original_stdout

#         # Write the captured output to a file
#         with open("smile_ouput.txt", 'w') as file:
#             file.write(output_buffer.getvalue())
#             file.write(error_buffer.getvalue())


#         # Close the StringIO object
#         output_buffer.close()    
#         error_buffer.close()
    
    # mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
    valid = [0 if mol is None else 1 for mol in mols]
    valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
    valid_mols = [mols[idx] for idx in valid_idxs]
    return valid_mols, valid_idxs


class Descriptors:

    def __init__(self, data):
        self._data = data

    def ECFP(self, radius, nBits):
        fingerprints = []
        mols, idx = smiles_to_mols(self._data)
        fp_bits = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]
        for fp in fp_bits:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, idx

    def ECFP_counts(self, radius, useFeatures, useCounts=True):
        mols, valid_idx = smiles_to_mols(self._data)
        fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp, valid_idx

    def Avalon(self, nBits):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=nBits) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

    def MACCS_keys(self):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, ), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

def get_ECFP6_counts(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.ECFP_counts(radius=3, useFeatures=True, useCounts=True)
    return fps

def get_model_and_val(args):
    model = None
    val = pd.DataFrame()
    if args.model_url:
        model_path = StorageManager.get_local_copy(args.model_url)
        with open(model_path, 'rb') as file:
          model = pickle.load(file)
    if args.val_url:
        val_url = StorageManager.get_local_copy(args.val_url)
        val = pd.read_csv(val_url, compression='gzip')
    return model, val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chain two Python programs that handle CSV input/output.")
    parser.add_argument("--model_url", required=False, default='', help="model url")
    parser.add_argument("--val_url", required=False, default='', help="validation data url")
    args = parser.parse_args()
    task = Task.init(project_name='paper', task_name='svm_val')
    model, val = get_model_and_val(args)
    if model is None:
        raise ValueError('model is None')
    val_fps = get_ECFP6_counts(val["SMILES"])
    y_pred = model.predict(X=val_fps)
    test_score = roc_auc_score(y_true=val["AV_Bit"], y_score=y_pred)
    print(test_score)
    print('DONE')
    task.close()
