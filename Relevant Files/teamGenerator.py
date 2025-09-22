####### IMPORTANT #######
# Claude AI helped a lot in generate code to calculate team scores and features. It did not include everything,
# So I spent a while going through and adjusting weights, fixing bugs, and adding comments. I do not know a lot
# about basketball, so having claude help seemed reasonable. Additionally, The output seems to actually make sense.



import pandas as pd
import numpy as np
from itertools import combinations
import random
from typing import List, Dict, Tuple
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

class BasketballTeamEvaluator:
    """
    Evaluates basketball teams based on basketball knowledge and principles.
    Scores teams on balance, complementary skills, and overall strength.
    """
    
    def __init__(self):
        # The weights determine how important certain features are. I played around with these to get different
        # scores and these seems to be effective values.
        self.weights = {
            'balance': 0.15, # General and physical stats: player_height, player_weigt, gp
            'strength': 0.15, # Personal stats: ts_pct, pts, reb, net_rating, oreb, dreb
            'complementary': 0.70 # Team interaction stats: ast, usg_pct, ast_pct
        }
        {# 1. player_name — Name of the player 
        # 2. team_abbreviation — Abbreviated name of the team the player played for (at the end of the season) -
        # 3. age — Age of the player -
        # 4. player_height — Height of the player (in centimeters)-
        # 5. player_weight — Weight of the player (in kilograms)-
        # 6. gp — Games played throughout the season-
        # 7. pts — Average number of points scored-
        # 8. reb — Average number of rebounds grabbed-
        # 9. ast — Average number of assists distributed
        # 10. net_rating — Team's point differential per 100 possessions while the player is on the court-
        # 11. oreb_pct — Percentage of available offensive rebounds the player grabbed while he was on the floor
        # 12. dreb_pct — Percentage of available defensive rebounds the player grabbed while he was on the floor
        # 13. usg_pct — Percentage of team plays used by the player while he was on the floor ((FGA + Possession Ending FTA + TO) / POSS)
        # 14. ts_pct — Measure of the player's shooting efficiency that takes into account free throws, 2 and 3 point shots (PTS / (2*(FGA + 0.44 * FTA)))
        # 15. ast_pct — Percentage of teammate field goals the player assisted while he was on the floor
        # 16. season — NBA season -
        }
        
    
    def evaluate_team(self, team_df: pd.DataFrame) -> Dict:
        """
        Evaluate a 5-player team and return detailed scoring.
        
        Args:
            team_df: DataFrame with 5 players and their stats
            
        Returns:
            Dictionary with scores and team features
        """
        
        # Calculate component scores
        balance_score = self._calculate_balance_score(team_df)
        strength_score = self._calculate_strength_score(team_df)
        complementary_score = self._calculate_complementary_score(team_df)
        
        # Overall weighted score
        total_score = (
            balance_score * self.weights['balance'] +
            strength_score * self.weights['strength'] +
            complementary_score * self.weights['complementary']
        )
        
        # Generate team features for neural network
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
        """Calculate how balanced the team is across different roles."""
        score = 0.0
        
        ### Height balance
        heights = team_df['player_height'].values
        height_std = np.std(heights)
        height_balance = min(height_std / 15.0, 1.0)  # Normalize by typical std
        score += height_balance * 0.4 # importance of the height feature

        ### Weight balance - variety in physicality
        weights = team_df['player_weight'].values
        weight_std = np.std(weights)
        weight_balance = min(weight_std / 20.0, 1.0)
        score += weight_balance * 0.4

        ### Games played - prefer players who played significant minutes
        games_played = team_df['gp']
        gp_score = np.mean(games_played) / 82.0   # linear normalization
        score += gp_score * 0.2
        
        return score
    
    def _calculate_strength_score(self, team_df: pd.DataFrame) -> float:
        """Calculate overall team strength in key areas, including rebounding and games played."""
        score = 0.0

        # ------------------------------
        # Offensive efficiency (True Shooting %)
        avg_ts = team_df['ts_pct'].fillna(0.5).mean()
        ts_score = min(max((avg_ts - 0.45) / 0.15, 0), 1)  # Scale 0.45-0.6 → 0-1
        score += ts_score * 0.1  # slightly lower to leave room for other stats

        # Scoring punch
        avg_pts = team_df['pts'].mean()
        pts_score = min(avg_pts / 20.0, 1.0)  # normalize to ~20 PPG
        score += pts_score * 0.3

         # Total rebounds
        avg_reb = team_df['reb'].fillna(0).mean()
        reb_score = min(avg_reb / 10.0, 1.0)  # normalize, assume 10 rpg is excellent
        score += reb_score * 0.15

        # Rebounding efficiency
        # Offensive rebounds
        oreb = team_df['oreb_pct'].fillna(0).mean()  # average oreb%
        oreb_score = min(oreb / 0.3, 1.0)  # assume 30% is elite
        score += oreb_score * 0.15

        # Defensive rebounds
        dreb = team_df['dreb_pct'].fillna(0).mean()  # average dreb%
        dreb_score = min(dreb / 0.8, 1.0)  # assume 80% is elite
        score += dreb_score * 0.15

        # Overall impact (net rating)
        avg_net = team_df['net_rating'].fillna(0).mean()
        net_score = min(max((avg_net + 10) / 20, 0), 1)  # Scale -10 → +10 to 0-1
        score += net_score * 0.15

        return score

    def _calculate_complementary_score(self, team_df: pd.DataFrame) -> float:
        """Calculate how well players complement each other based on usage and playmaking."""

        score = 0.0

        # ------------------------------
        # Usage rate balance: avoid 5 ball-dominant players
        usage_rates = team_df['usg_pct'].fillna(0.15).values
        avg_usage = np.mean(usage_rates)
        usage_balance = 1.0 - min(abs(avg_usage - 0.22) / 0.1, 1.0)  # Optimal ~22%
        score += usage_balance * 0.4  # weight for usage balance

        # ------------------------------
        # Playmaking distribution: ensure some players handle passing
        playmakers_pct = sum(team_df['ast_pct'].fillna(0.1) > 0.2)  # high assist %
        playmaker_balance_pct = min(playmakers_pct / 2.0, 1.0)      # want ~1-2 playmakers
        score += playmaker_balance_pct * 0.3  # weight for ast_pct

        # ------------------------------
        # Total assists: reward teams with more passing overall
        total_ast = team_df['ast'].fillna(0).sum()
        # Normalize: assume 20-40 assists per game is good for a 5-player team
        ast_score = min(total_ast / 40.0, 1.0)
        score += ast_score * 0.3  # weight for raw assists

        return score

    
    def _generate_team_features(self, team_df: pd.DataFrame) -> List[float]:
        """Generate numerical features for neural network input."""
        features = []
        
        # Basic averaged stats
        features.extend([
            team_df['pts'].mean(),
            team_df['reb'].mean(), 
            team_df['ast'].mean(),
            team_df['ts_pct'].fillna(0.5).mean(),
            team_df['net_rating'].fillna(0).mean(),
            team_df['usg_pct'].fillna(0.15).mean(),
            team_df['ast_pct'].fillna(0.1).mean(),
        ])
        
        # Variability measures (balance)
        features.extend([
            team_df['pts'].std(),
            team_df['reb'].std(),
            team_df['player_height'].std(),
            team_df['age'].std(),
            team_df['usg_pct'].fillna(0.15).std(),
        ])
        
        # Team composition features
        features.extend([
            team_df['player_height'].max() - team_df['player_height'].min(),  # Height range
            team_df['age'].max() - team_df['age'].min(),  # Age range
            sum(team_df['pts'] > 15),  # High scorers
            sum(team_df['ast'] > 4),   # Playmakers  
            sum(team_df['reb'] > 7),   # Strong rebounders
            sum(team_df['ts_pct'].fillna(0.4) > 0.55),  # Good shooters
        ])
        
        # Physical attributes
        features.extend([
            team_df['player_height'].mean(),
            team_df['player_weight'].mean(),
            team_df['age'].mean(),
        ])
        
        return features

def generate_teams_and_scores(csv_file="relevant_data.csv", sample_size=5000, random_seed=42):
    """
    Generate random team combinations and score them.
    
    Args:
        csv_file: Path to cleaned player data
        sample_size: Number of random teams to generate and score
        random_seed: For reproducible results
    """
    # Load player data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} players from {csv_file}")
    
    # Initialize evaluator
    evaluator = BasketballTeamEvaluator()
    
    # Generate random team combinations
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    team_data = []
    feature_data = []
    scores = []
    
    print(f"Generating and scoring {sample_size} random teams...")
    
    for i in range(sample_size):
        # Randomly select 5 players
        team_indices = random.sample(range(len(df)), 5)
        team_df = df.iloc[team_indices].copy()
        
        # Evaluate team
        evaluation = evaluator.evaluate_team(team_df)
        
        # Store results
        team_data.append({
            'team_id': i,
            'players': evaluation['players'],
            'total_score': evaluation['total_score'],
            'balance_score': evaluation['balance_score'],
            'strength_score': evaluation['strength_score'],
            'complementary_score': evaluation['complementary_score']
        })
        
        feature_data.append(evaluation['features'])
        scores.append(evaluation['total_score'])
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{sample_size} teams")
    
    # Convert to DataFrames
    teams_df = pd.DataFrame(team_data)
    
    # Create features DataFrame
    feature_names = [
        'avg_pts', 'avg_reb', 'avg_ast', 'avg_ts_pct', 'avg_net_rating', 'avg_usg_pct', 'avg_ast_pct',
        'std_pts', 'std_reb', 'std_height', 'std_age', 'std_usg_pct', 
        'height_range', 'age_range', 'high_scorers', 'playmakers', 'rebounders', 'good_shooters',
        'avg_height', 'avg_weight', 'avg_age'
    ]
    
    features_df = pd.DataFrame(feature_data, columns=feature_names)
    features_df['team_score'] = scores
    
    # Save results
    teams_df.to_csv(os.path.join(script_dir,'team_evaluations.csv'), index=False)
    features_df.to_csv(os.path.join(script_dir,'team_features.csv'), index=False)
    
    print(f"\nResults saved:")
    print(f"  - team_evaluations.csv: Team compositions and scores")
    print(f"  - team_features.csv: Features for neural network training")
    print(f"\nScore distribution:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std:  {np.std(scores):.3f}")
    print(f"  Min:  {np.min(scores):.3f}")
    print(f"  Max:  {np.max(scores):.3f}")
    
    # Show best team
    best_idx = np.argmax(scores)
    print(f"\nBest team (Score: {scores[best_idx]:.3f}):")
    for player in team_data[best_idx]['players']:
        print(f"  - {player}")
    
    return teams_df, features_df

if __name__ == "__main__":
    # Generate teams and scores
    input_file=os.path.join(script_dir, "relevant_data.csv")

    teams_df, features_df = generate_teams_and_scores(
        csv_file=input_file,
        sample_size=5000  # Start with 5000 teams, can increase later
    )