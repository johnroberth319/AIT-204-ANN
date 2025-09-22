#### WORK FLOW ####
# Run these in order:
# 1. dataCleaner.py: clean and filter raw data. Outputs to relevant data.
# 2. teamGenerator.py: creates random teams from players based on stats.
# 3. ANN.py: defines neural network class for training and prediction.
# 4. MAIN.py: main script to run the full pipeline. Finds the most optimal team. from a lot of different combinations.

import ANN
import os
import pandas as pd
from itertools import combinations
from teamGenerator import BasketballTeamEvaluator
import streamlit as st
from tensorflow import keras

script_dir = os.path.dirname(os.path.abspath(__file__))

### FUNCTION FOR FINDING OPTIMAL TEAM ###
def find_optimal_team(csv_file="relevant_data.csv", model=None, optional_max_teams=500):
    """
    Find the optimal team using the trained neural network.
    
    Args:
        csv_file: Path to player data
        model: Trained BasketballNeuralNetwork instance
    
    Returns:
        Best team DataFrame and score
    """
    # Load player data
    df = pd.read_csv(os.path.join(script_dir, csv_file))
    print(f"Finding optimal team from {len(df)} players...")
    
    evaluator = BasketballTeamEvaluator() # used to evaluate teams.
    
    best_score = 0
    best_team = None
    best_features = None
    
    # Generate all possible 5-player combinations
    # This will generate over 75,000,000 combinations for 100 players. There is no way our computers can handle that,
    # so fall backs have been implemented, such as only using 500 teams.
    total_combinations = len(list(combinations(range(len(df)), 5)))
    st.write(f"Total possible teams: {total_combinations:,}")
    
    if total_combinations > 100000:
        st.write(f"Too many combinations! Sampling {optional_max_teams} random teams...")
        # Sample random combinations
        import random
        random.seed(42)
        team_combinations = [random.sample(range(len(df)), 5) for _ in range(optional_max_teams)]
    else:
        print("Evaluating all possible combinations...")
        team_combinations = list(combinations(range(len(df)), 5))
    
    st.write(f"Evaluating {len(team_combinations):,} teams...")
    
    for i, team_indices in enumerate(team_combinations):
        # Get team
        team_df = df.iloc[list(team_indices)]
        
        # Generate features
        features = evaluator._generate_team_features(team_df)
        
        # Predict score
        predicted_score = model.predict_team_score(features)
        
        # Track best team
        if predicted_score > best_score:
            best_score = predicted_score
            best_team = team_df.copy()
            best_features = features
        
        # Progress update
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(team_combinations):,} teams. Best score so far: {best_score:.4f}")
    
    print(f"\nOPTIMAL TEAM FOUND!")
    print(f"Predicted Score: {best_score:.4f}/1")
    print(f"\nPlayers:")
    for idx, player in best_team.iterrows():
        print(f"  • {player['player_name']}")
    
    # Save optimal team (not needed but nice to have)
    best_team.to_csv(os.path.join(script_dir, 'optimal_team.csv'), index=False)
    # print(f"\nOptimal team saved to 'optimal_team.csv'")
    
    return best_team, best_score

def main():
    MODEL_PATH = os.path.join(script_dir, "basketball_team_model.keras")

    st.write("SEARCHING FOR OPTIMAL TEAM...")

    # Always initialize your wrapper class
    nn = ANN.NeuralNetwork(input_dim=21)

    # If the .keras model file exists, load it into the wrapper
    if os.path.exists(MODEL_PATH):
        st.write("Loading existing model from .keras file...")
        nn.load_model()  # This loads into your wrapper class
    else:
        # Build and train new model
        nn.build_model(hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001)
        
        X_train, X_test, y_train, y_test = nn.prepare_data(features_file="team_features.csv", test_size=0.2)
        
        nn.train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=64, patience=10)
        
        nn.plot_training_history()
        nn.save_model()

    # Now nn is always your wrapper class with predict_team_score()
    optimal_team, optimal_score = find_optimal_team(
        csv_file="relevant_data.csv",
        model=nn  # This now has predict_team_score()
    )

    return nn, optimal_team, optimal_score


if __name__ == "__main__":
    # Run the complete pipeline
    st.title("🏀 Optimal Basketball Team Finder")

    st.markdown("""
                # GitHub Repository
                [Link to GitHub Repository](https://github.com/nolanrd04/CST-435-Assignment2-ANNv1.1)
                I wasn't sure if you wanted the whole thing submitted in a zip file, or just the streamlit link.
                # AI usage
                Claude AI helped a lot in generating code to calculate team scores and features. It did not include everything,
                So I spent a while going through and adjusting weights, fixing bugs, and adding comments. I do not know a lot
                about basketball, so having claude help seemed reasonable. Additionally, The output seems to actually make sense.
                Both ChatGPT and Claude were used in helping to create the pipeline and clean things up.

                # Project explanation

                ### This project uses basketball player data to build the most optimal team of 5 players using a neural network.
                ### Here are the steps involved:
                1. **Data Cleaning**: The raw player data is cleaned and filtered to keep relevant statistics. Only 100 players are picked.
                2. **Team Feature Generation**: Random teams are generated and their features are calculated. Picked from the 100 players.
                3. **Rules established**: There are rules made in order to determine what makes a good team.
                There are three categories that each have their own weights, and they are comprised of individual
                features that also have their own weights. The weights of the featres were determined by my general
                knowledge of basketball and trial and error. The weights of the categories were determined by trial
                and error.
                4. **Neural Network Training**: A neural network is built and trained to predict team quality scores based on the generated features.
                The neural network learns from the diffeerent combiations of teams and their scores.
                5. **Optimal Team Search**: The trained model is used to evaluate many possible 5-player combinations to find the best team, given the 100 players.
                """)
    
    model, best_team, score = main()
    
    st.subheader("Best Team Found")
    st.write(f"Predicted Score: {score:.4f}/1")
    st.dataframe(best_team)