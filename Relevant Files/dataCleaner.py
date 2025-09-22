import pandas as pd
import random
import os

# ==============================
# Data Cleaning Function
# ==============================

def clean_data(
    input_file="all_seasons.csv",
    output_file="relevant_data.csv",
    start_year=None,
    num_players=100
):
    # Dynamic path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(input_file):
        input_file = os.path.join(script_dir, input_file)
    if not os.path.isabs(output_file):
        output_file = os.path.join(script_dir, output_file)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"--- Error: {input_file} not found!")
        print("Please download the NBA dataset from Kaggle and place it in the same directory as this script.")
        return False
    
    # Load dataset
    df = pd.read_csv(input_file)
    print(f"1. Loaded dataset with {len(df)} rows")

    # Extract start year from season column (e.g., "1996-97" → 1996)
    df["season_start"] = df["season"].str.split("-").str[0].astype(int)

    # If using a 5-year window
    if start_year is not None:
        end_year = start_year + 4
    else:
        # Pick random valid start year if none provided
        max_year = df["season_start"].max()
        min_year = df["season_start"].min()
        start_year = random.randint(min_year, max_year - 4)
        end_year = start_year + 4
        print(f"2. Randomly selected start year: {start_year}")

    # Filter rows within the 5-year window
    df = df[(df["season_start"] >= start_year) & (df["season_start"] <= end_year)]
    print(f"3. Filtered to {start_year}-{end_year}: {len(df)} rows")

    # Keep only relevant columns
    relevant_columns = [
        "player_name",
        "team_abbreviation",
        "age",
        "player_height",
        "player_weight",
        "gp",
        "pts",
        "reb",
        "ast",
        "net_rating",
        "oreb_pct",
        "dreb_pct",
        "usg_pct",
        "ts_pct",
        "ast_pct",
        "season"
    ]
    
    # Check which columns exist in the dataset
    available_columns = [col for col in relevant_columns if col in df.columns]
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    
    if missing_columns:
        print(f"---  Warning: Missing columns: {missing_columns}")
        print(f"+++ Using available columns: {available_columns}")
    
    df = df[available_columns]

    # Remove rows with missing critical data
    critical_columns = ["player_name", "pts", "reb", "ast", "player_height"]
    available_critical = [col for col in critical_columns if col in df.columns]
    
    initial_count = len(df)
    df = df.dropna(subset=available_critical)
    print(f"4. Removed {initial_count - len(df)} rows with missing critical data")

    # Randomly sample players (default 100)
    if num_players is not None and num_players < len(df):
        df = df.sample(n=num_players, random_state=42)  # fixed seed for reproducibility
        print(f"5. Randomly sampled {num_players} players")

    # Save filtered dataset
    df.to_csv(output_file, index=False)
    print(f"6. Cleaned data ({start_year}-{end_year}, {len(df)} players) saved to {os.path.basename(output_file)}")
    return True




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
clean_data(
        input_file=os.path.join(script_dir, "all_seasons.csv"),
        output_file=os.path.join(script_dir, "relevant_data.csv")
    )
    