import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec


model = word2vec.Word2Vec.load('examples//models//model_300dim.pkl')

df = pd.read_csv('drug_smiles.csv')


def smiles_to_vec(smiles, model, radius=1, unseen='UNK'):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * model.vector_size
    sentence = mol2alt_sentence(mol, radius)
    mol_sentence = MolSentence(sentence)
    mol_vec = DfVec(sentences2vec([mol_sentence], model, unseen=unseen)[0])
    return mol_vec.vec

tqdm.pandas(desc="Extracting mol2vec features")
df['features'] = df['smiles'].progress_apply(lambda smi: smiles_to_vec(smi, model))

features_df = pd.DataFrame(df['features'].tolist(), columns=[f'f_{i}' for i in range(model.vector_size)])


mol2vec_feature_df = pd.concat([df[['drug_id']], features_df], axis=1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

features_df_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
mol2vec_feature_standard = pd.concat([df[['drug_id']], features_df_scaled], axis=1)
mol2vec_feature_standard.to_csv('drugbank_mol2vec_standard.csv', index=False)
