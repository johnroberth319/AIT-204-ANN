# streamlit_app.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from typing import List, Dict
from tensorflow import keras
import math

# ==========================================
# Neural Network Wrapper (Load Only)
# ==========================================
class NeuralNetwork:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None

    def load_model(self, path="basketball_team_model.keras"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        self.model = keras.models.load_model(path)

    def predict_team_score(self, features: List[float]) -> float:
        features = np.array(features).reshape(1, -1)
        return float(self.model.predict(features, verbose=0)[0][0])

# ==========================================
# Basketball Team Evaluator
# ==========================================
class BasketballTeamEvaluator:
    def _generate_team_features(self, team_df) -> List[float]:
        features = []
        features.extend([
            team_df["pts"].mean(),
            team_df["reb"].mean(),
            team_df["ast"].mean(),
            team_df["ts_pct"].fillna(0.5).mean(),
            team_df["net_rating"].fillna(0).mean(),
            team_df["usg_pct"].fillna(0.15).mean(),
            team_df["ast_pct"].fillna(0.1).mean(),
        ])
        features.extend([
            team_df["pts"].std(),
            team_df["reb"].std(),
            team_df["player_height"].std(),
            team_df["age"].std(),
            team_df["usg_pct"].fillna(0.15).std(),
        ])
        features.extend([
            team_df["player_height"].max() - team_df["player_height"].min(),
            team_df["age"].max() - team_df["age"].min(),
            sum(team_df["pts"] > 15),
            sum(team_df["ast"] > 4),
            sum(team_df["reb"] > 7),
            sum(team_df["ts_pct"].fillna(0.4) > 0.55),
        ])
        features.extend([
            team_df["player_height"].mean(),
            team_df["player_weight"].mean(),
            team_df["age"].mean(),
        ])
        return features

# ==========================================
# Optimal Team Finder (Load CSV + Model Only)
# ==========================================
def find_optimal_team(csv_file="relevant_data.csv", model=None, optional_max_teams=500):
    if not os.path.exists(csv_file):
        st.error(f"❌ Missing data file: {csv_file}")
        st.stop()

    df = pd.read_csv(csv_file)
    evaluator = BasketballTeamEvaluator()

    best_score, best_team = 0, None

    total_combinations = math.comb(len(df), 5)
    st.write(f"Total possible teams: {total_combinations:,}")


    if total_combinations > 100000:
        st.write(f"Too many combinations! Sampling {optional_max_teams} random teams...")
        team_combinations = [random.sample(range(len(df)), 5) for _ in range(optional_max_teams)]
    else:
        team_combinations = list(combinations(range(len(df)), 5))

    for team_indices in team_combinations:
        team_df = df.iloc[list(team_indices)]
        features = evaluator._generate_team_features(team_df)
        predicted_score = model.predict_team_score(features)
        if predicted_score > best_score:
            best_score, best_team = predicted_score, team_df.copy()

    return best_team, best_score

# ==========================================
# Streamlit UI
# ==========================================
st.title("🏀 Optimal Basketball Team Finder")

MODEL_PATH = "basketball_team_model.keras"
nn = NeuralNetwork(input_dim=21)

try:
    nn.load_model(MODEL_PATH)
    st.success(f"✅ Loaded model: {MODEL_PATH}")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if st.button("Find Best Team"):
    best_team, score = find_optimal_team("relevant_data.csv", model=nn)
    st.subheader("Best Team Found")
    st.write(f"Predicted Score: {score:.4f}/1")
    st.dataframe(best_team)
