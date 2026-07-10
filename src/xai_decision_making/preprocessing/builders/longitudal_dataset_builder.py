"""
Longitudinal dataset construction for ConDOR ordinal regression.

Merges PACAP QoL questionnaire data with NKR clinical records, computes
QLQ-C30 summary scores, and produces padded time-series sequences per
patient as PyTorch DataLoaders.
"""
from xai_decision_making import config
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from xai_decision_making.preprocessing.qol.qlq_c30_scoring import calculate_qlq_c30_scores

class LongitudinalDataset(Dataset):
    def __init__(self, sequences, max_len=10):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        actual_len = min(len(seq["t"]), self.max_len)
        pad_len = self.max_len - actual_len

        x = torch.FloatTensor(np.pad(seq["x"][:actual_len], (0, pad_len)))
        t = torch.FloatTensor(
            np.pad(seq["t"][:actual_len], (0, pad_len), constant_values=-1)
        )
        y = torch.LongTensor(np.pad(seq["y"][:actual_len], (0, pad_len)))
        mask = torch.BoolTensor(np.pad(np.ones(actual_len), (0, pad_len)))

        static = torch.FloatTensor(seq["static"])

        treat_data = seq["treat"][:actual_len]
        treat_pad = np.zeros((pad_len, treat_data.shape[1]))
        treat = torch.FloatTensor(np.concatenate([treat_data, treat_pad], axis=0))

        return x, t, y, mask, static, treat

class LongitudinalDataFactory:
    def __init__(self, pacap_path, clinical_path, include_treatment=True):
        self.pacap_path = pacap_path
        self.clinical_path = clinical_path
        self.include_treatment = include_treatment

        self.ID_COL = config.KEY_NK
        self.TIME_COL = config.PROMS_DATE_COL
        self.LABEL_COL = "stadium"

        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.static_dim = 0
        self.static_matrix = None

    def load_and_process(self):
        df_pacap = pd.read_csv(self.pacap_path)
        df_clinical = pd.read_csv(self.clinical_path)

        scores = calculate_qlq_c30_scores(df_pacap)
        df_pacap["qol_summary"] = scores["Summary Score: QLQ-C30 Summary Score"] / 100.0

        df_merged = pd.merge(df_pacap, df_clinical, on=self.ID_COL, how="left")

        df_merged[self.TIME_COL] = pd.to_numeric(
            df_merged[self.TIME_COL], errors="coerce"
        )

        df_merged["time_months"] = df_merged[self.TIME_COL] / 30.44
        df_merged["time_normalized"] = df_merged["time_months"] / 60.0

        df_clean = df_merged.dropna(subset=["time_normalized", "qol_summary"]).copy()

        return df_clean

    def create_sequences(self, df_clean):
        sequences = []
        grouped = df_clean.sort_values([self.ID_COL, "time_normalized"]).groupby(
            self.ID_COL
        )

        for patient_id, group in grouped:
            if len(group) < 2:
                continue

            sequences.append(
                {
                    "t": group["time_normalized"].values,
                    "x": group["qol_summary"].values,
                    "y": group[self.LABEL_COL].values,
                    "static": np.zeros(1),
                    "treat": np.zeros((len(group), 6)),
                }
            )

        return sequences

    def get_loaders(self, batch_size=32):
        df = self.load_and_process()
        sequences = self.create_sequences(df)

        dataset = LongitudinalDataset(sequences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return loader
