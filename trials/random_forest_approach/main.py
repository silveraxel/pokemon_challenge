import json
import pandas as pd
from tqdm.notebook import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


DATA_PATH = "."
TRAIN_FILE_PATH = os.path.join(DATA_PATH, 'train.jsonl')
TEST_FILE_PATH = os.path.join(DATA_PATH, 'test.jsonl')
TEST_SIZE = 0.3


def main():

    transformers = [
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), 
         ["p1_main_type", "p2_lead_type1", "p2_lead_type2"])
    ]
    ct = ColumnTransformer(transformers, remainder='passthrough')

    # Load and prepare training data
    dataset_json = load_dataset(TRAIN_FILE_PATH)
    logging.info(f"Number of training rows: {len(dataset_json)}")

    dataset = extract_data(dataset_json, is_test_data=False)
    y = dataset['player_won']
    X = dataset.drop('player_won', axis=1)

    # Fit transformer only on training data
    X_transformed = ct.fit_transform(X)

    # Train/Validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=42
    )

    # Train model on training split
    model = RandomForestClassifier(n_estimators=100, max_depth=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate on validation
    p_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, p_test)
    logging.info(f'Accuracy test {acc_test}')

    # Retrain on full training data
    model.fit(X_transformed, y)

    # Load and process test data
    test_json = load_dataset(TEST_FILE_PATH)
    test_df = extract_data(test_json, is_test_data=True)

    # Apply the same transformation to the test data
    test_df_transformed = ct.transform(test_df)

    # Generate predictions
    test_predictions = model.predict(test_df_transformed)

    # Create submission
    submission_df = pd.DataFrame({
        'battle_id': [battle['battle_id'] for battle in test_json],
        'player_won': test_predictions
    })

    submission_df.to_csv('submission.csv', index=False)
    logging.info("Submission file 'submission.csv' created successfully!")



def extract_average_p1_team_value(p1_team):
    """
    Calculate average stats for player 1's team and extract main type
    """
    # Initialize stat totals
    total_hp = 0
    total_atk = 0
    total_def = 0
    total_spa = 0
    total_spd = 0
    total_spe = 0
    total_lev = 0
    
    # type counting
    type_counts = {}
    
    # Sum up stats for each Pokemon and count types
    for pokemon in p1_team:
        total_hp += pokemon["base_hp"]
        total_atk += pokemon["base_atk"]
        total_def += pokemon["base_def"]
        total_spa += pokemon["base_spa"]
        total_spd += pokemon["base_spd"]
        total_spe += pokemon["base_spe"]
        total_lev += pokemon["level"]
        for t in pokemon.get("types", []):
            if t == "notype":
                continue
            type_counts[t] = type_counts.get(t, 0) + 1
    
    # Determine main type (most frequent), fallback 'unknown'
    if type_counts:
        p1_main_type = max(type_counts.items(), key=lambda x: x[1])[0]
    else:
        p1_main_type = "unknown"
    
    # Calculate averages
    team_size = len(p1_team)
    return {
        "p1_team_avg_hp": total_hp / team_size,
        "p1_team_avg_atk": total_atk / team_size,
        "p1_team_avg_def": total_def / team_size,
        "p1_team_avg_spa": total_spa / team_size,
        "p1_team_avg_spd": total_spd / team_size,
        "p1_team_avg_spe": total_spe / team_size,
        "p1_team_avg_lev": total_lev / team_size,
        # categorical for one-hot
        "p1_main_type": p1_main_type
    }

def extract_data(dataset_json, is_test_data=False):
    dataset = []

    for battle in dataset_json:
        row = {}
        
        # Base feature
        if not is_test_data:
            row["player_won"] = int(battle["player_won"])
        
        # Add team average stats
        row.update(extract_average_p1_team_value(battle["p1_team_details"]))
        
        # Add p2 lead stats
        row.update(extract_p2_lead_stats(battle["p2_lead_details"]))
        
        # Add final Pokemon states
        row.update(extract_final_pokemon_stats(battle["battle_timeline"]))
        
        # Add timeline features
        row.update(extract_timeline_features(battle["battle_timeline"]))
        
        dataset.append(row)
    return pd.DataFrame(dataset).fillna(0)

def extract_p2_lead_stats(p2_lead):
    """
    Extract base stats for player 2's lead Pokemon (including types for OHE)
    """
    p2_lead_type1, p2_lead_type2 = p2_lead.get("types", [])
    
    
    return {
        "p2_lead_hp": p2_lead["base_hp"],
        "p2_lead_atk": p2_lead["base_atk"],
        "p2_lead_def": p2_lead["base_def"],
        "p2_lead_spa": p2_lead["base_spa"],
        "p2_lead_spd": p2_lead["base_spd"],
        "p2_lead_spe": p2_lead["base_spe"],
        "p2_lead_lev": p2_lead["level"],
        "p2_lead_type1": p2_lead_type1,
        "p2_lead_type2": p2_lead_type2
    }

def extract_final_pokemon_stats(battle_timeline):
    """
    Extract the last state of each Pokemon in the battle and calculate average stats

    """
    if not battle_timeline:
        return {
            "p1_final_avg_hp": 0.0,
            "p2_final_avg_hp": 0.0,
            "p1_final_avg_boosts": 0.0,
            "p2_final_avg_boosts": 0.0,
            "p1_final_status_count": 0,
            "p2_final_status_count": 0,
            "p1_final_effects_count": 0,
            "p2_final_effects_count": 0,
            "p1_fainted_count": 0,
            "p2_fainted_count": 0
        }

    # Dictionary to store the last state of each Pokemon
    p1_final_states = {}
    p2_final_states = {}
    
    # Scan the timeline in reverse to get last state of each Pokemon
    for turn in reversed(battle_timeline):
        p1_state = turn.get("p1_pokemon_state", {})
        p2_state = turn.get("p2_pokemon_state", {})
        p1_name = p1_state.get("name")
        p2_name = p2_state.get("name")
        
        if p1_name and p1_name not in p1_final_states:
            p1_final_states[p1_name] = {
                "hp_pct": p1_state.get("hp_pct", 0.0),
                "status": p1_state.get("status", "nostatus"),
                "effects": p1_state.get("effects", []),
                "boosts": sum(v for v in p1_state.get("boosts", {}).values() if isinstance(v, (int, float)))
            }
        if p2_name and p2_name not in p2_final_states:
            p2_final_states[p2_name] = {
                "hp_pct": p2_state.get("hp_pct", 0.0),
                "status": p2_state.get("status", "nostatus"),
                "effects": p2_state.get("effects", []),
                "boosts": sum(v for v in p2_state.get("boosts", {}).values() if isinstance(v, (int, float)))
            }

    # Safe averages
    def safe_avg(values):
        return sum(values) / len(values) if values else 0.0

    p1_avg_hp = safe_avg([s["hp_pct"] for s in p1_final_states.values()])
    p2_avg_hp = safe_avg([s["hp_pct"] for s in p2_final_states.values()])

    p1_avg_boosts = safe_avg([s["boosts"] for s in p1_final_states.values()])
    p2_avg_boosts = safe_avg([s["boosts"] for s in p2_final_states.values()])

    # Count non-normal status conditions
    p1_final_status = sum(1 for s in p1_final_states.values() if s["status"] not in ["nostatus", "noeffect"])
    p2_final_status = sum(1 for s in p2_final_states.values() if s["status"] not in ["nostatus", "noeffect"])

    # Count effects: consider an effect as present if effects list contains any value != "noeffect"
    def has_meaningful_effect(effects_list):
        return any(e != "noeffect" for e in (effects_list or []))

    p1_final_effects = sum(1 for s in p1_final_states.values() if has_meaningful_effect(s.get("effects")))
    p2_final_effects = sum(1 for s in p2_final_states.values() if has_meaningful_effect(s.get("effects")))

    p1_fainted = sum(1 for s in p1_final_states.values() if s["status"] == "fnt")
    p2_fainted = sum(1 for s in p2_final_states.values() if s["status"] == "fnt")
    
    return {
        "p1_final_avg_hp": p1_avg_hp,
        "p2_final_avg_hp": p2_avg_hp,
        "p1_final_avg_boosts": p1_avg_boosts,
        "p2_final_avg_boosts": p2_avg_boosts,
        "p1_final_status_count": p1_final_status,
        "p2_final_status_count": p2_final_status,
        "p1_final_effects_count": p1_final_effects,
        "p2_final_effects_count": p2_final_effects,
        "p1_fainted_count": p1_fainted,
        "p2_fainted_count": p2_fainted
    }

def extract_timeline_features(timeline):
    """
    Estrarre poche feature utili dalla timeline:
      - avg base_power delle mosse danno per p1/p2
      - count mosse di tipo DAMAGE vs STATUS
      - count specifica di 'recover' (semplice proxy per cura)
    """
    def safe_get_move(m): 
        return m or {}
    p1_powers = []
    p2_powers = []
    p1_damage_count = 0
    p2_damage_count = 0
    p1_status_count = 0
    p2_status_count = 0
    p1_recover = 0
    p2_recover = 0

    for turn in (timeline or []):
        m1 = safe_get_move(turn.get("p1_move_details"))
        m2 = safe_get_move(turn.get("p2_move_details"))

        bp1 = m1.get("base_power")
        if isinstance(bp1, (int, float)) and bp1 > 0:
            p1_powers.append(bp1); p1_damage_count += 1
        elif m1.get("category") == "STATUS":
            p1_status_count += 1
        if m1.get("name", "").lower() == "recover":
            p1_recover += 1

        bp2 = m2.get("base_power")
        if isinstance(bp2, (int, float)) and bp2 > 0:
            p2_powers.append(bp2); p2_damage_count += 1
        elif m2.get("category") == "STATUS":
            p2_status_count += 1
        if m2.get("name", "").lower() == "recover":
            p2_recover += 1

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0

    return {
        "p1_avg_move_power": avg(p1_powers),
        "p2_avg_move_power": avg(p2_powers),
        "p1_damage_move_count": p1_damage_count,
        "p2_damage_move_count": p2_damage_count,
        "p1_status_move_count": p1_status_count,
        "p2_status_move_count": p2_status_count,
        "p1_recover_count": p1_recover,
        "p2_recover_count": p2_recover,
    }



def load_dataset(file_name) -> dict:
    dataset_json = []

    with open(file_name, 'r') as f:
        for line in f:
            dataset_json.append(json.loads(line))

    return dataset_json



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()







