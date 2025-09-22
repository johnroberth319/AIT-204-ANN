# streamlit_app.py
import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from tensorflow import keras
import math

# ==========================================
# Neural Network Wrapper
# ==========================================
class NeuralNetwork:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = None
        self.scaler = None  # add scaler for consistent preprocessing

    def load_model(self, model_path="basketball_team_model.keras", scaler_path="feature_scaler.pkl"):
        """Load trained model and scaler."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ Scaler file not found: {scaler_path}")
        
        # Load model
        self.model = keras.models.load_model(model_path)

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Scaler loaded from {scaler_path}")

    def predict_team_score(self, features: List[float]) -> float:
        """Predict team score given feature vector."""
        if self.model is None or self.scaler is None:
            raise ValueError("❌ Model or scaler not loaded. Call load_model() first.")

        # Ensure correct shape
        features = np.array(features).reshape(1, -1)

        # Scale features like during training
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled, verbose=0)[0][0]
        return float(prediction)

# ==========================================
# Basketball Team Evaluator
# ==========================================
class BasketballTeamEvaluator:
    def __init__(self):
        self.weights = {
            'balance': 0.15,
            'strength': 0.15,
            'complementary': 0.70
        }

    def evaluate_team(self, team_df: pd.DataFrame) -> Dict:
        balance_score = self._calculate_balance_score(team_df)
        strength_score = self._calculate_strength_score(team_df)
        complementary_score = self._calculate_complementary_score(team_df)

        total_score = (
            balance_score * self.weights['balance'] +
            strength_score * self.weights['strength'] +
            complementary_score * self.weights['complementary']
        )

        features = self._generate_team_features(team_df)

        return {
            'total_score': total_score,
            'balance_score': balance_score,
            'strength_score': strength_score,
            'complementary_score': complementary_score,
            'features': features,
            'players': team_df['player_name'].tolist()
        }

    def _calculate_balance_score(self, team_df: pd.DataFrame) -> float:
        score = 0.0
        height_std = np.std(team_df['player_height'].values)
        score += min(height_std / 15.0, 1.0) * 0.4
        weight_std = np.std(team_df['player_weight'].values)
        score += min(weight_std / 20.0, 1.0) * 0.4
        gp_score = np.mean(team_df['gp']) / 82.0
        score += gp_score * 0.2
        return score

    def _calculate_strength_score(self, team_df: pd.DataFrame) -> float:
        score = 0.0
        avg_ts = team_df['ts_pct'].fillna(0.5).mean()
        score += min(max((avg_ts - 0.45) / 0.15, 0), 1) * 0.1
        score += min(team_df['pts'].mean() / 20.0, 1.0) * 0.3
        score += min(team_df['reb'].fillna(0).mean() / 10.0, 1.0) * 0.15
        score += min(team_df['oreb_pct'].fillna(0).mean() / 0.3, 1.0) * 0.15
        score += min(team_df['dreb_pct'].fillna(0).mean() / 0.8, 1.0) * 0.15
        avg_net = team_df['net_rating'].fillna(0).mean()
        score += min(max((avg_net + 10) / 20, 0), 1) * 0.15
        return score

    def _calculate_complementary_score(self, team_df: pd.DataFrame) -> float:
        score = 0.0
        usage_rates = team_df['usg_pct'].fillna(0.15).values
        avg_usage = np.mean(usage_rates)
        score += (1.0 - min(abs(avg_usage - 0.22) / 0.1, 1.0)) * 0.4
        playmakers = sum(team_df['ast_pct'].fillna(0.1) > 0.2)
        score += min(playmakers / 2.0, 1.0) * 0.3
        total_ast = team_df['ast'].fillna(0).sum()
        score += min(total_ast / 40.0, 1.0) * 0.3
        return score

    def _generate_team_features(self, team_df: pd.DataFrame) -> List[float]:
        features = []
        features.extend([
            team_df['pts'].mean(),
            team_df['reb'].mean(),
            team_df['ast'].mean(),
            team_df['ts_pct'].fillna(0.5).mean(),
            team_df['net_rating'].fillna(0).mean(),
            team_df['usg_pct'].fillna(0.15).mean(),
            team_df['ast_pct'].fillna(0.1).mean(),
        ])
        features.extend([
            team_df['pts'].std(),
            team_df['reb'].std(),
            team_df['player_height'].std(),
            team_df['age'].std(),
            team_df['usg_pct'].fillna(0.15).std(),
        ])
        features.extend([
            team_df['player_height'].max() - team_df['player_height'].min(),
            team_df['age'].max() - team_df['age'].min(),
            sum(team_df['pts'] > 15),
            sum(team_df['ast'] > 4),
            sum(team_df['reb'] > 7),
            sum(team_df['ts_pct'].fillna(0.4) > 0.55),
        ])
        features.extend([
            team_df['player_height'].mean(),
            team_df['player_weight'].mean(),
            team_df['age'].mean(),
        ])
        return features

# ==========================================
# Optimal Team Finder
# ==========================================
def find_optimal_team(csv_file="relevant_data.csv", model=None, optional_max_teams=500):
    if not os.path.exists(csv_file):
        st.error(f"❌ Missing data file: {csv_file}")
        st.stop()

    df = pd.read_csv(csv_file)
    evaluator = BasketballTeamEvaluator()

    best_score, best_team, best_eval, best_model_score = 0, None, None, 0

    total_combinations = math.comb(len(df), 5)
    st.write(f"Total possible teams: {total_combinations:,}")

    if total_combinations > 100000:
        st.write(f"Too many combinations! Sampling {optional_max_teams} random teams...")
        team_combinations = [random.sample(range(len(df)), 5) for _ in range(optional_max_teams)]
    else:
        team_combinations = list(combinations(range(len(df)), 5))

    for team_indices in team_combinations:
        team_df = df.iloc[list(team_indices)]
        evaluation = evaluator.evaluate_team(team_df)
        model_score = model.predict_team_score(evaluation['features'])
        if evaluation['total_score'] > best_score:
            best_score = evaluation['total_score']
            best_team = team_df.copy()
            best_eval = evaluation
            best_model_score = model_score

    return best_team, best_score, best_eval, best_model_score

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
    best_team, score, eval_details, model_score = find_optimal_team("relevant_data.csv", model=nn)
    st.subheader("Best Team Found")
    st.write(f"🏆 Evaluation Score: {score:.4f}/1")
    st.write(f"🤖 Model Prediction: {model_score:.4f}/1")

    st.markdown("### Breakdown of Scores")
    st.write({
        "Balance": f"{eval_details['balance_score']:.4f}",
        "Strength": f"{eval_details['strength_score']:.4f}",
        "Complementary": f"{eval_details['complementary_score']:.4f}",
    })

    st.markdown("### Players")
    st.dataframe(best_team)
