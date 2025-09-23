import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.preprocessing import StandardScaler

tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-smiles2caption")
encoder = T5EncoderModel.from_pretrained("laituan245/molt5-large-smiles2caption")
encoder.eval()

df = pd.read_csv('drug_smiles.csv')

def extract_molt5_features(smiles):
    try:
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = encoder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        #print(f"Feature shape for {smiles}: {embedding.shape}")

        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing SMILES: {smiles}\n{e}")
        return None

tqdm.pandas(desc="Extracting MolT5 features")

df['features'] = df['smiles'].progress_apply(extract_molt5_features)
df['features'] = df['features'].apply(lambda x: x.tolist() if x is not None else x)

df = df.dropna(subset=['features'])

feature_df = df['features'].apply(pd.Series)
feature_df.columns = [f"f_{i}" for i in range(feature_df.shape[1])]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_df)

feature_df_scaled = pd.DataFrame(features_scaled, columns=feature_df.columns)

output_scaled_df = pd.concat([df['drug_id'].reset_index(drop=True), feature_df_scaled], axis=1)

output_scaled_df.to_csv('drugbank_molt5_standard.csv', index=False)


