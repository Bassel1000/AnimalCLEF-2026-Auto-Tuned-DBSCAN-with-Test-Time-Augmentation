# AnimalCLEF 2026: Auto-Tuned DBSCAN with Test-Time Augmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-AnimalCLEF_2026-skyblue)

This repository contains an optimized solution for the **[AnimalCLEF 2026](https://www.kaggle.com/competitions/animal-clef-2026/)** competition on Kaggle. 

The goal of the competition is to cluster images of individual animals (Lynx, Salamanders, Sea Turtles, and Horned Lizards) to re-identify unique individuals without explicit ID labels in the test set.

## 🚀 Solution Overview

This solution improves upon the standard baseline by removing manual guesswork from the clustering parameters and adding robustness to the feature extraction process.

### Key Techniques

1.  **Automatic Hyperparameter Tuning (DBSCAN `eps`)**
    * Instead of using arbitrary density thresholds (e.g., `eps=0.5`), this notebook utilizes the **Training Set** to find the mathematical optimum.
    * It runs a grid search on the training data for each species, calculating the **Adjusted Rand Index (ARI)** to find the exact `eps` value that maximizes clustering accuracy.
    * *Result:* Species-specific thresholds that adapt to the density of the embedding space for that particular animal.

2.  **Test-Time Augmentation (TTA)**
    * Animal re-identification models can be sensitive to orientation.
    * We extract features for both the original image and a **horizontally flipped** version.
    * The embeddings are averaged to create a robust feature vector that is less sensitive to pose noise.

3.  **State-of-the-Art Feature Extractors**
    * **Lynx & Horned Lizards:** `conservationxlabs/miewid-msv3` (EfficientNetV2 backbone).
    * **Salamanders & Sea Turtles:** `hf-hub:BVRA/MegaDescriptor-L-384` (MegaDescriptor).

## 📂 Repository Structure

```text
├── animalclef-2026-auto-tuned-dbscan-with-test-time.ipynb   # Main notebook
├── README.md                                                # Project documentation
└── submission.csv                                           # Generated submission file
```
## 🛠️ Requirements

The solution relies on the `wildlife-tools` and `wildlife-datasets` libraries. If running locally, install the dependencies:

```bash
pip install torch torchvision timm transformers scikit-learn pandas tqdm
pip install wildlife-datasets wildlife-tools
```
*Note: In the Kaggle environment, you may need to enable "Internet" access to download the pre-trained models from Hugging Face.*

## ⚙️ How It Works

The notebook proceeds in three main stages:

### 1. Data Loading & Preprocessing
* Loads the dataset using `wildlife_datasets`.
* Splits data into species-specific subsets (`LynxID2025`, `SalamanderID2025`, etc.).

### 2. Tuning Stage (The "Smart" Part)
* Iterates through the **Training** split of each species.
* Extracts features and computes a cosine similarity matrix.
* Tests `eps` values from `0.05` to `0.60`.
* Selects the `eps` that yields the highest ARI score against ground truth labels.

### 3. Inference Stage
* Applies the **Optimized `eps`** values to the **Test** split.
* Performs DBSCAN clustering.
* Formats the output into the required `cluster_{species}_{id}` format for Kaggle submission.

## 📊 Performance Notes

* **Baseline `eps` (Guesswork):** Often leads to under-clustering (merging different animals) or over-clustering (splitting one animal into many).
* **Tuned `eps`:** Align the clustering density with the actual separation capabilities of the model for that specific species.

## 🤝 Credits

* **Competition:** [Kaggle AnimalCLEF 2026](https://www.kaggle.com/competitions/animal-clef-2026/)
* **Libraries:** [Wildlife Datasets](https://github.com/WildlifeDatasets/wildlife-datasets) & [Wildlife Tools](https://github.com/WildlifeDatasets/wildlife-tools)
* **Models:** * [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384)
    * [MiewId](https://github.com/ConservationX-Labs/miewid)

---
*Created for educational and competitive purposes in the AnimalCLEF 2026 challenge.*
