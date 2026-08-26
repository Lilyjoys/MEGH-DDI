
## **MEGH-DDI:Edge-Aware Graph Learning with Hop2Token Representations for Drug–Drug Interaction Prediction Using Large Language Models**

This project implements a framework for **drug–drug interaction (DDI) prediction**, combining **edge-aware graph learning** with **Hop2Token molecular representations**.It supports **DrugBank** and **ChChMiner** datasets under both **inductive** and **transductive** settings, with ready-to-run training scripts.

---

## ⚙️ Environment Setup

```bash
# Create environment
conda create -n ddi python=3.7
conda activate ddi

# Install PyTorch + CUDA 11.3
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia

# Core dependencies
conda install numpy=1.21.5 scipy=1.7.3 scikit-learn=1.0.2 matplotlib seaborn pandas
```


## 📂 Dataset Configuration

Switch between datasets by modifying the **`dataset_name`** field in `arg.json`.  
Available options include:

- `miner_s2_1`
- `miner_t`
- `drugbank_t_2`
- `drugbank_s1_0`


## 🚀 Training Scripts

### DrugBank
```bash
# Inductive
python train_drugbank_induc_loss.py

# Transductive
python train_drugbank_trans_loss.py
```
### Miner
```bash
# Inductive
python train_miner_induc_loss.py

# Transductive
python train_miner_trans_loss.py
```



