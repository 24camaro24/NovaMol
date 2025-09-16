
# 🧪 NOVAMOL

This project combines **Graph Neural Networks (GNNs)** and **Recurrent Neural Networks (RNNs)** to explore the chemical space of molecules. It is capable of:

* 🧬 **Generating novel molecules** from SMILES strings using an LSTM-based RNN.
* 🔗 **Building molecular graphs** (atoms as nodes, bonds as edges) with PyTorch Geometric.
* ⚡ **Predicting molecular properties** (e.g., dipole moment) using a trained GNN.
* 🌍 **Identifying novelty** of generated molecules by comparing fingerprints (Tanimoto similarity).
* 📊 **Validating predictions** against known PubChem molecules and computing accuracy.
* 💡 **Exploring industrial applications** of discovered molecules (future direction).

---

## 🚀 Project Workflow

1. **Dataset Preparation**

   * Uses QM9/QM40 dataset (`main.csv`) containing molecules with atomic coordinates, bonds, and properties like dipole moment.
   * PubChem API is queried to verify if generated molecules already exist.

2. **Graph Representation**

   * Atoms → nodes (encoded with atomic number `Z`).
   * Bonds → edges (with bond order as edge attributes).
   * Built using `torch_geometric.data.Data`.

3. **Modeling**

   * **RNN (LSTM)** → generates novel molecules in SMILES form.
   * **GNN** → predicts dipole moments from molecular graphs.

4. **Novelty Check**

   * Morgan fingerprints + Tanimoto similarity.
   * If similarity < 0.7 → considered "novel molecule."

5. **Evaluation**

   * Predictions validated against **PubChem** experimental values (where available).
   * Accuracy computed per molecule and averaged.

---

## 📂 Repository Structure

```
📦 molecule-gnn-rnn
├── data/
│   ├── main.csv          # QM9/QM40 dataset
│   ├── bonds.csv         # Bond info
│   ├── xyz.csv           # Atomic coordinates
├── models/
│   ├── gnn_model.py      # Graph Neural Network for property prediction
│   ├── rnn_model.py      # LSTM model for SMILES generation
├── notebooks/
│   ├── train_gnn.ipynb   # Training pipeline for GNN
│   ├── generate_rnn.ipynb# Molecule generation via RNN
│   ├── validate.ipynb    # PubChem validation & accuracy
├── utils/
│   ├── fingerprints.py   # Morgan fingerprint + similarity
│   ├── pubchem_api.py    # PubChem data fetching
│   ├── graph_utils.py    # Build PyG graphs from dataset
├── README.md
└── requirements.txt
```

---

## 📊 Results

* Predicted dipole moments of generated molecules.
* Verified subset with PubChem → **average prediction accuracy \~75%**.
* Identified **novel molecules** with no PubChem entries.

---

## 🔮 Future Work

* Map **predicted properties → industrial applications**.
* Extend beyond dipole moment to properties like polarizability, solubility, HOMO-LUMO gap.
* Deploy as a **web app demo** (similar to AlphaFold DB).

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/molecule-gnn-rnn.git
cd molecule-gnn-rnn
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train GNN

```bash
python train_gnn.py
```

### Generate Molecules with RNN

```bash
python generate_rnn.py
```

### Validate Against PubChem

```bash
python validate.py
```

---

## 📚 References

* QM9 / QM40 Dataset
* PubChem API
* PyTorch Geometric
* RDKit for chemical informatics

---

## 👥 Contributors

* **wasae** 
* **raja** 
* **john** 
* **hari** 


