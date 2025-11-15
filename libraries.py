
import pandas as pd
import json
import os
from toolbox import *
import sys

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np




# **Creation of the Dataframes, to work on it**.
# 
# We used 4 different data frames to better manage the data and we divided them in this way: 
# * df_battle, data on the winner;
# * df_squad, team of p1;
# * df_pokemon, starting pokemon of p2;
# * df_battle_timeline, battle divided into rounds. 
# 
# In df_battle_timeline is used p1_/p2_ to represent data of the first player/second player

def create_dataframe(data):
    # DataFrame with only the information about the winner
    df_battle = pd.DataFrame([{
        "battle_id": b["battle_id"],
        "player_won": b.get("player_won", None)#for the test is None
    } for b in data])

    # DataFrame with the Pokémons from the first squad (p1_team_details)
    df_squad = pd.DataFrame([
        {
            "battle_id": b["battle_id"],
            "pokemon_name": p["name"],
            "level": p["level"],
            "types": p["types"],
            "base_hp": p["base_hp"],
            "base_atk": p["base_atk"],
            "base_def": p["base_def"],
            "base_spa": p["base_spa"],
            "base_spd": p["base_spd"],
            "base_spe": p["base_spe"]
        }
        for b in data
        for p in b["p1_team_details"]
    ])

    # DataFrame with the details of the lead Pokémon from second squas
    df_pokemon = pd.DataFrame([
        {
            "battle_id": b["battle_id"],
            "name": b["p2_lead_details"]["name"],
            "level": b["p2_lead_details"]["level"],
            "types": b["p2_lead_details"]["types"],
            "base_hp": b["p2_lead_details"]["base_hp"],
            "base_atk": b["p2_lead_details"]["base_atk"],
            "base_def": b["p2_lead_details"]["base_def"],
            "base_spa": b["p2_lead_details"]["base_spa"],
            "base_spd": b["p2_lead_details"]["base_spd"],
            "base_spe": b["p2_lead_details"]["base_spe"]
        }
        for b in data
    ])

    # DataFrame with the timeline of our battle
    df_battle_timeline = pd.DataFrame([
        {
            "battle_id": b["battle_id"],
            "turn": t["turn"],
            "p1_pokemon": t["p1_pokemon_state"]["name"],
            "p1_hp": t["p1_pokemon_state"]["hp_pct"],
            "p1_status": t["p1_pokemon_state"]["status"],
            "p1_effects": t["p1_pokemon_state"]["effects"],
            "p1_boosts": t["p1_pokemon_state"]["boosts"],
            "p2_pokemon": t["p2_pokemon_state"]["name"],
            "p2_hp": t["p2_pokemon_state"]["hp_pct"],
            "p2_status": t["p2_pokemon_state"]["status"],
            "p2_effects": t["p2_pokemon_state"]["effects"],
            "p2_boosts": t["p2_pokemon_state"]["boosts"],
            "p1_move_name": t["p1_move_details"]["name"] if t["p1_move_details"] else None,
            "p1_move_type": t["p1_move_details"]["type"] if t["p1_move_details"] else None,
            "p1_move_cat": t["p1_move_details"]["category"] if t["p1_move_details"] else None,
            "p1_move_basepow": t["p1_move_details"]["base_power"] if t["p1_move_details"] else None,
            "p1_move_acc": t["p1_move_details"]["accuracy"] if t["p1_move_details"] else None,
            "p1_move_priority": t["p1_move_details"]["priority"] if t["p1_move_details"] else None,
            "p2_move_name": t["p2_move_details"]["name"] if t["p2_move_details"] else None,
            "p2_move_type": t["p2_move_details"]["type"] if t["p2_move_details"] else None,
            "p2_move_cat": t["p2_move_details"]["category"] if t["p2_move_details"] else None,
            "p2_move_basepow": t["p2_move_details"]["base_power"] if t["p2_move_details"] else None,
            "p2_move_acc": t["p2_move_details"]["accuracy"] if t["p2_move_details"] else None,
            "p2_move_priority": t["p2_move_details"]["priority"] if t["p2_move_details"] else None
        }
        for b in data
        for t in b["battle_timeline"]
    ])

    return [df_battle, df_squad, df_pokemon, df_battle_timeline]



# # **FEATURE EXTRACTION** of stats before the match

# From the dataframes (df) df_pokemon and df_squad we obtain the basic stats of the Pokémon. For team p1 we take the average of these values, which will then be reduced based on the value of the lead Pokémon of p2. Also, for each Pokémon present in the battle we will see what type it is and for each player, p1 and p2, we will count the occurrences of these types. For example, p1_fire=1 if p1 has only 1 fire-type Pokémon.





def extract_feature_diff(lista,isTrain):
    df_battle = lista[0]
    df_pokemon = lista[2]
    df_squad = lista[1]

    unique_types = unique_t(lista)

    # --- PLAYER 1 ---
    df_squad["types_clean"] = df_squad["types"].apply(lambda x: [t for t in x if t != "notype"])
    agg_squad1 = df_squad.groupby("battle_id").agg({
        "base_hp": "mean",
        "base_atk": "mean",
        "base_def": "mean",
        "base_spa": "mean",
        "base_spd": "mean",
        "base_spe": "mean",
        "level": "mean",
        "types_clean": lambda lst: [t for sub in lst for t in sub]
    }).reset_index()

    for t in unique_types:
        agg_squad1[t] = agg_squad1["types_clean"].apply(lambda lst: lst.count(t))

    agg_squad1 = (
        agg_squad1
        .drop(columns=["types_clean"])
        .add_prefix("p1_")
        .rename(columns={"p1_battle_id": "battle_id"})
    )

    # --- PLAYER 2 ---
    df_squad2 = df_pokemon.copy()
    df_squad2["types_clean"] = df_squad2["types"].apply(lambda x: [t for t in x if t != "notype"])
    for t in unique_types:
        df_squad2[t] = df_squad2["types_clean"].apply(lambda lst: lst.count(t))

    df_squad2 = df_squad2.drop(columns=["types_clean", "types", "name"], errors="ignore")
    agg_squad2 = df_squad2.add_prefix("p2_").rename(columns={"p2_battle_id": "battle_id"})

    agg_full = agg_squad1.merge(agg_squad2, on="battle_id", how="inner")
    agg_full = agg_full.merge(df_battle[["battle_id", "player_won"]], on="battle_id", how="left")
    
    if isTrain:
        agg_full["player_won"] = agg_full["player_won"].astype(int)
    else:
        agg_full["player_won"] = None
    # Final features are the difference between p1_stats - p2_stats
    base_stats = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe", "level"]
    for stat in base_stats:
        col_p1 = f"p1_{stat}"
        col_p2 = f"p2_{stat}"
        diff_col = f"diff_{stat}"
        agg_full[diff_col] = agg_full[col_p1] - agg_full[col_p2]

    # To remove original features
    cols_to_drop = [f"p1_{s}" for s in base_stats] + [f"p2_{s}" for s in base_stats]
    agg_full = agg_full.drop(columns=cols_to_drop)
    return agg_full.fillna(0).infer_objects()




# Very similar to the previous one but it will extract features by not only looking at data from the p1 squad and the lead pokemon from p2, like extract_feature_diff(), but also retrieving the types of squad p2 using the battle timeline and the pokedex created. This means having more information about the squad p2


def extract_feature_diff_tottipi(lista,isTrain):
    df_battle = lista[0]
    df_pokemon = lista[2]
    df_squad = lista[1]

    unique_types = unique_t(lista)

    # --- PLAYER 1 ---
    df_squad["types_clean"] = df_squad["types"].apply(lambda x: [t for t in x if t != "notype"])
    agg_squad1 = df_squad.groupby("battle_id").agg({
        "base_hp": "mean",
        "base_atk": "mean",
        "base_def": "mean",
        "base_spa": "mean",
        "base_spd": "mean",
        "base_spe": "mean",
        "level": "mean",
        "types_clean": lambda lst: [t for sub in lst for t in sub]
    }).reset_index()
    for t in unique_types:
        agg_squad1[t] = agg_squad1["types_clean"].apply(lambda lst: lst.count(t))
    agg_squad1 = (
        agg_squad1
        .drop(columns=["types_clean"])
        .add_prefix("p1_")
        .rename(columns={"p1_battle_id": "battle_id"})
    )

    # --- PLAYER 2 --
    df_squad2 = df_pokemon.copy()
    df_squad2["types_clean"] = df_squad2["types"].apply(lambda x: [t for t in x if t != "notype"])
    for t in unique_types:
        df_squad2[t] = 0.0

    df_squad2 = df_squad2.drop(columns=["types_clean", "types", "name"], errors="ignore")
    agg_squad2 = df_squad2.add_prefix("p2_").rename(columns={"p2_battle_id": "battle_id"})

    
    agg_full = agg_squad1.merge(agg_squad2, on="battle_id", how="inner")
    agg_full = agg_full.merge(df_battle[["battle_id", "player_won"]], on="battle_id", how="left")
    
    if isTrain:
        agg_full["player_won"] = agg_full["player_won"].astype(int)
    else:
        agg_full["player_won"] = None
    
    # Final features are the difference between p1_stats - p2_stats
    base_stats = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe", "level"]
    for stat in base_stats:
        col_p1 = f"p1_{stat}"
        col_p2 = f"p2_{stat}"
        diff_col = f"diff_{stat}"
        agg_full[diff_col] = agg_full[col_p1] - agg_full[col_p2]

    # To remove original features 
    cols_to_drop = [f"p1_{s}" for s in base_stats] + [f"p2_{s}" for s in base_stats]
    agg_full = agg_full.drop(columns=cols_to_drop)
    # Difference from the previous function
    agg_full=tipiSquadra2(agg_full,unique_types,lista) 
    return agg_full.fillna(0).infer_objects()


# Unlike the two previous functions, here we'll also extract features from the df_battle_timeline. As you can see from the code, we called extract_feature_diff() for the features from before the match that were already good.
# In this function, we calculate the difference between p1 and p2 for the following parameters: hp, movebasepow, moveacc, and boost obtained during the match. We also save the occurrences, for each player, of the various statuses, effects, and move types that could affect pokemons because they can became game changer.


def extract_all(lista,isTrain):
    unique_status,unique_effects=unique_se(lista)
    static_features=extract_feature_diff(lista,isTrain)
    df_battle_timeline = lista[3]

    #Extract dynamic feature
    dynamic_features = df_battle_timeline.groupby('battle_id').agg({
        'p1_hp': 'mean', 
        'p2_hp': 'mean',
        'p1_move_basepow': 'mean',
        'p2_move_basepow': 'mean',
        'p1_move_acc': 'mean',
        'p2_move_acc': 'mean'
    }).reset_index()
    
    # p1 - p2
    dynamic_features['hp_diff_mean'] = dynamic_features['p1_hp'] - dynamic_features['p2_hp']
    dynamic_features['move_basepow_diff_mean'] = dynamic_features['p1_move_basepow'] - dynamic_features['p2_move_basepow']
    dynamic_features['move_acc_diff_mean'] = dynamic_features['p1_move_acc'] /dynamic_features['p2_move_acc']


    # Remove original features
    dynamic_features = dynamic_features.drop(columns=[
        'p1_hp', 'p2_hp',
        'p1_move_basepow', 'p2_move_basepow',
        'p1_move_acc', 'p2_move_acc'
    ])
    
    # life left( minimum of 0 to 6 because 1 means 100% of hp left)
    tot_life=extract_hp_sum(df_battle_timeline)

    # Status 
    p1_status = status_counts(df_battle_timeline, 'p1',unique_status)
    p2_status = status_counts(df_battle_timeline, 'p2',unique_status)

    # Effects
    p1_effects = effects_counts(df_battle_timeline, 'p1',unique_effects)
    p2_effects = effects_counts(df_battle_timeline, 'p2',unique_effects)

    # Move type and categories
    p1_moves = move_type_counts(df_battle_timeline, 'p1')
    p2_moves = move_type_counts(df_battle_timeline, 'p2')

    # Boosts
    df_boosts=boosts_sum(df_battle_timeline)

    # MERGE ALL DATASET
    dynamic_dfs = [
        dynamic_features,
        tot_life,
        p1_status,
        p2_status,
        p1_effects,
        p2_effects,
        df_boosts,
        p1_moves, 
        p2_moves
    ]
    dynamic_full = dynamic_dfs[0]
    for extra_df in dynamic_dfs[1:]:
        dynamic_full = dynamic_full.merge(extra_df, on='battle_id', how='left')
    
    # Merge features of the before match  
    train_df = (
        static_features
        .merge(dynamic_full, on='battle_id', how='left')
    )
    print(train_df.columns)
    return train_df.fillna(0).infer_objects()


# Similar function to the extract_all() but this time use ragg_status() and ragg_effects(), so it doesn't use the singular effects and status but some macro-categories, like p1_effect_trap =p1_effect_clamp+p1_effect_firespin+p1_effect_wrap, because create similar situation to the pokemon.



def extract_all_ragg(lista, is_train):
    unique_status,unique_effects=unique_se(lista)
    static_features=extract_feature_diff(lista, is_train)
    df_battle_timeline = lista[3]

    #Extract dynamic feature
    dynamic_features = df_battle_timeline.groupby('battle_id').agg({
        'p1_hp': 'mean', 
        'p2_hp': 'mean',
        'p1_move_basepow': 'mean',
        'p2_move_basepow': 'mean',
        'p1_move_acc': 'mean',
        'p2_move_acc': 'mean'
    }).reset_index()
    
    # p1 - p2
    dynamic_features['hp_diff_mean'] = dynamic_features['p1_hp'] - dynamic_features['p2_hp']
    dynamic_features['move_basepow_diff_mean'] = dynamic_features['p1_move_basepow'] - dynamic_features['p2_move_basepow']
    dynamic_features['move_acc_diff_mean'] = dynamic_features['p1_move_acc'] /dynamic_features['p2_move_acc']


    # Remove original features
    dynamic_features = dynamic_features.drop(columns=[
        'p1_hp', 'p2_hp',
        'p1_move_basepow', 'p2_move_basepow',
        'p1_move_acc', 'p2_move_acc'
    ])
    
    # life left
    tot_life=extract_hp_sum(df_battle_timeline)

    # Status 
    p_status = ragg_status(df_battle_timeline,unique_status) 

    # Effects
    p_effects = ragg_effects(df_battle_timeline,unique_effects)    

    # Move type and categories
    p1_moves = move_type_counts(df_battle_timeline, 'p1')
    p2_moves = move_type_counts(df_battle_timeline, 'p2')

    # Boosts
    df_boosts=boosts_sum(df_battle_timeline)

    # MERGE ALL DATASET
    dynamic_dfs = [
        dynamic_features,
        tot_life,
        p_status,
        p_effects,
        df_boosts,
        p1_moves, 
        p2_moves
    ]
    dynamic_full = dynamic_dfs[0]
    for extra_df in dynamic_dfs[1:]:
        dynamic_full = dynamic_full.merge(extra_df, on='battle_id', how='left')
    
    # Merge features of the before match 
    train_df = (
        static_features
        .merge(dynamic_full, on='battle_id', how='left')
    )

    train_df["somma_diff_hp"] = (
        train_df["somma_diff_hp"] - train_df["somma_diff_hp"].mean()
    ) / train_df["somma_diff_hp"].std()
    print(train_df.columns)
    return train_df.fillna(0).infer_objects()

# A variation of extract_all_ragg() where in this case it's used extract_feature_diff_tottipi() and not extract_feature_diff(). There aren't any difference other thant this one. 

# In[ ]:


#Extraction of the features using also the battletimeline to recreate the p2 teams before the match



def extract_all_tottipi(lista, is_train):
    unique_status,unique_effects=unique_se(lista)
    static_features=extract_feature_diff_tottipi(lista, is_train)
    df_battle_timeline = lista[3]

    #Extract dynamic feature
    dynamic_features = df_battle_timeline.groupby('battle_id').agg({
        'p1_hp': 'mean', 
        'p2_hp': 'mean',
        'p1_move_basepow': 'mean',
        'p2_move_basepow': 'mean',
        'p1_move_acc': 'mean',
        'p2_move_acc': 'mean'
    }).reset_index()
    
    # p1 - p2
    dynamic_features['hp_diff_mean'] = dynamic_features['p1_hp'] - dynamic_features['p2_hp']
    dynamic_features['move_basepow_diff_mean'] = dynamic_features['p1_move_basepow'] - dynamic_features['p2_move_basepow']
    dynamic_features['move_acc_diff_mean'] = dynamic_features['p1_move_acc'] /dynamic_features['p2_move_acc']


    # Removed original features
    dynamic_features = dynamic_features.drop(columns=[
        'p1_hp', 'p2_hp',
        'p1_move_basepow', 'p2_move_basepow',
        'p1_move_acc', 'p2_move_acc'
    ])
    
    # life left
    tot_life=extract_hp_sum(df_battle_timeline)

    # Status 
    p_status = ragg_status(df_battle_timeline,unique_status) 

    # Effects 
    p_effects = ragg_effects(df_battle_timeline,unique_effects)    

    # Move type and categories
    p1_moves = move_type_counts(df_battle_timeline, 'p1')
    p2_moves = move_type_counts(df_battle_timeline, 'p2')

    # Boosts
    df_boosts=boosts_sum(df_battle_timeline)

    # MERGE DATASET
    dynamic_dfs = [
        dynamic_features,
        tot_life,
        p_status,
        p_effects,
        df_boosts,
        p1_moves, 
        p2_moves
    ]
    dynamic_full = dynamic_dfs[0]
    for extra_df in dynamic_dfs[1:]:
        dynamic_full = dynamic_full.merge(extra_df, on='battle_id', how='left')
    
    # Merge features of the before match 
    train_df = (
        static_features
        .merge(dynamic_full, on='battle_id', how='left')
    )

   
    train_df["somma_diff_hp"] = (
        train_df["somma_diff_hp"] - train_df["somma_diff_hp"].mean()
    ) / train_df["somma_diff_hp"].std()
    return train_df.fillna(0).infer_objects()


