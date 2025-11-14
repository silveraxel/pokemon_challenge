import pandas as pd

def unique_se(lista):
    df_battle_timeline=lista[3]
    # For status
    all_status = pd.concat([
        df_battle_timeline['p1_status'],
        df_battle_timeline['p2_status']
    ], ignore_index=True)
    unique_status = (
        all_status.dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    unique_status = (
        all_status.dropna()
        .astype(str)
        .unique()
        .tolist()
    )   
    unique_status.remove("nostatus")
    print("status unici:", unique_status)

    # For effects
    all_effects = []
    for col in ["p1_effects", "p2_effects"]:
        for row in df_battle_timeline[col].dropna():
            if isinstance(row, list):
                all_effects.extend(row)
            elif isinstance(row, str):
                all_effects.append(row)

    unique_effects = sorted(set(all_effects))
    unique_effects.remove("noeffect")
    print("EFFECTS unici:", unique_effects)
    return unique_status,unique_effects


#Function to obtain all the pokemon's type(UNICI)
def unique_t(lista):
    df_pokemon=lista[2]
    df_squad=lista[1]
    unique_types = sorted(
        set(
            t
            for types_list in pd.concat([df_pokemon["types"], df_squad["types"]])
            for t in types_list
        )
    )
    return unique_types


#Function to count the occurrence of each status for each player during the battle
def status_counts(df, prefix,unique):
  counts = (
      df.groupby(['battle_id', f'{prefix}_status'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
  )
  counts = counts[['battle_id'] + unique]
  #Add status to the dataframe df
  for st in unique:
    if st not in counts.columns:
        counts[st] = 0
  #prefix can be p1 or p2
  counts.columns = ['battle_id'] + [f'{prefix}_status_{c}' for c in counts.columns if c != 'battle_id']
  return counts



#Function to count the occurrence of each effects for each player during the battle
from collections import Counter
import pandas as pd
def effects_counts(df, prefix,unique):
    rows = []
    for bid, group in df.groupby('battle_id'):
        all_effects = []
        for eff in group[f'{prefix}_effects'].dropna():
            if isinstance(eff, list):
                all_effects.extend(eff)
            elif isinstance(eff, str):
                all_effects.append(eff)
        counts = Counter(all_effects)
        rows.append({'battle_id': bid, **counts})

    eff_df = pd.DataFrame(rows).fillna(0)

    # Add possible effects not observed during the battle
    for eff in unique:
        if eff not in eff_df.columns:
            eff_df[eff] = 0
    eff_df = eff_df[['battle_id'] + unique]

    # prefix can be p1 or p2
    eff_df.columns = ['battle_id'] + [f'{prefix}_effect_{c}' for c in unique]
    return eff_df


#Function to count the occurrence of each move categories and types
def move_type_counts(df, prefix):
    #types
    type_counts = (
        df.groupby(['battle_id', f'{prefix}_move_type'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    type_counts.columns = ['battle_id'] + [f'{prefix}_movetype_{c}' for c in type_counts.columns if c != 'battle_id']

    #categories
    cat_counts = (
        df.groupby(['battle_id', f'{prefix}_move_cat'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    cat_counts.columns = ['battle_id'] + [f'{prefix}_movecat_{c}' for c in cat_counts.columns if c != 'battle_id']
    return type_counts.merge(cat_counts, on='battle_id', how='outer').fillna(0).infer_objects()



#Function to sum all the boosts obtained for each team during the battle
import pandas as pd
def boosts_sum(df):
    rows = []
    for bid, group in df.groupby('battle_id'):
        # initialization
        p1_tot = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
        p2_tot = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}

        for _, row in group.iterrows():
            if isinstance(row.get('p1_boosts'), dict):
                for k in p1_tot.keys():
                    p1_tot[k] += row['p1_boosts'].get(k, 0)
            if isinstance(row.get('p2_boosts'), dict):
                for k in p2_tot.keys():
                    p2_tot[k] += row['p2_boosts'].get(k, 0)

        # build a unique row for each battle
        rows.append({
            'battle_id': bid,
            **{f'p1_boost_{k}': v for k, v in p1_tot.items()},
            **{f'p2_boost_{k}': v for k, v in p2_tot.items()}
        })

    return pd.DataFrame(rows)



import pandas as pd
#Function that get the live left of each pokemon
def extract_hp_sum(df):

    def last_hp_sum(df, pokemon_col, hp_col):
        # Take last %hp for each pokemon
        last_hp = (
            df.dropna(subset=[pokemon_col])
              .sort_values(['battle_id', pokemon_col, 'turn'])
              .groupby(['battle_id', pokemon_col], as_index=False)
              .last()[['battle_id', hp_col]]
        )
        # Sum all percentage
        return last_hp.groupby('battle_id')[hp_col].sum().rename(f"somma_{pokemon_col[:2]}")

    somma_p1 = last_hp_sum(df, 'p1_pokemon', 'p1_hp')
    somma_p2 = last_hp_sum(df, 'p2_pokemon', 'p2_hp')

    result = pd.concat([somma_p1, somma_p2], axis=1).reset_index().fillna(0)
    
    #number of pokemon
    n_pokemon = (
        df.groupby('battle_id')
        .agg({
            'p1_pokemon': lambda x: len(set(x.dropna())),
            'p2_pokemon': lambda x: len(set(x.dropna()))
        })
        .rename(columns={
            'p1_pokemon': 'p1_pokemon_used',
            'p2_pokemon': 'p2_pokemon_used'
        })
        .reset_index()
    )

    result = result.merge(n_pokemon, on='battle_id', how='left')
    
    # Add percentage of pokemons not used
    result['somma_p1'] = result['somma_p1'] + (6 - result['p1_pokemon_used'])
    result['somma_p2'] = result['somma_p2'] + (6 - result['p2_pokemon_used'])
    result['somma_diff_hp']=result['somma_p1']-result['somma_p2']

    return result[['battle_id', 'somma_diff_hp']]



#Function to create a pokedex
def create_pokedex(lista_df):
    df_list=lista_df
    df_squad=df_list[1]
    df_pokemon=df_list[2]
    df_all_pokemon = pd.concat([df_squad, df_pokemon], ignore_index=True)
    df_all_pokemon["types"] = df_all_pokemon["types"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    df_all_pokemon = (
        df_all_pokemon.drop_duplicates(subset=["name", "types"])
        .reset_index(drop=True)
    )
    return df_all_pokemon



#Function that via battle timeline takes the type for each Pokémon of the second team
def tipiSquadra2(df,lista,lista_df):
    df_pokedex=create_pokedex(lista_df)[["name", "types"]]
    df_battle_timeline=lista_df[3]
    
    for tipo in lista:
        col = f"p2_{tipo}"
        if col not in df.columns:
            df[col] = 0.0
    
    #  Iterate on every battle
    for battle_id in df["battle_id"].unique():
        # Squad of p2
        p2_pokemon_list = (
            df_battle_timeline.loc[df_battle_timeline["battle_id"] == battle_id, "p2_pokemon"]
            .dropna()
            .unique()
            .tolist()
        )
        df_p2_pokemons = df_pokedex[df_pokedex["name"].isin(p2_pokemon_list)].copy()
        types_list = [t for types in df_p2_pokemons["types"] for t in types if t != "notype"]
        # Update df
        df.loc[df["battle_id"] == battle_id, "p2team_size"]=max(len(p2_pokemon_list), 1)

        for elem in types_list:
            col = f"p2_{elem}"
            df.loc[df["battle_id"] == battle_id, col] +=1
    return df



#Another version of the function tipiSquadra2()
def tipiSquadra22(df, unique_types, lista_df):
    df_pokedex = create_pokedex(lista_df)[["name", "types"]]
    df_battle_timeline = lista_df[3]

    p2_battles = (
        df_battle_timeline.groupby("battle_id")["p2_pokemon"]
        .apply(lambda x: list(set(x.dropna())))
        .reset_index()
    )

    p2_battles = p2_battles.explode("p2_pokemon").dropna(subset=["p2_pokemon"])

    merged = p2_battles.merge(df_pokedex, left_on="p2_pokemon", right_on="name", how="left")

    merged = merged.dropna(subset=["types"])
    exploded = (
        merged.explode("types")
        .groupby(["battle_id", "types"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for tipo in unique_types:
        if tipo not in exploded.columns:
            exploded[tipo] = 0
    exploded = exploded.add_prefix("p2_").rename(columns={"p2_battle_id": "battle_id"})
    df = df.merge(exploded, on="battle_id", how="left").fillna(0).infer_objects()

    return df



def ragg_status(dff,unique):
    p1_status = status_counts(dff, 'p1',unique)
    p2_status = status_counts(dff, 'p2',unique)
    df = p1_status.merge(p2_status, on='battle_id', how='left').fillna(0)
    expected_status = ['par', 'slp', 'frz', 'brn', 'tox', 'psn', 'fnt']
    for s in expected_status:
        for p in ['p1', 'p2']:
            col = f"{p}_status_{s}"
            if col not in df.columns:
                df[col] = 0
                
    # Macro-categories
    # STUNNED: par, slp, frz
    df["p1_status_stunned"] = df["p1_status_par"] + df["p1_status_slp"] + df["p1_status_frz"]
    df["p2_status_stunned"] = df["p2_status_par"] + df["p2_status_slp"] + df["p2_status_frz"]

    # POISONED: tox, psn
    df["p1_status_poisoned"] = df["p1_status_tox"] + df["p1_status_psn"]
    df["p2_status_poisoned"] = df["p2_status_tox"] + df["p2_status_psn"]

    # RESIDUAL_DMG: brn + (tox, psn)
    df["p1_status_dmgresiduo"] = df["p1_status_brn"] + df["p1_status_tox"] + df["p1_status_psn"]
    df["p2_status_dmgresiduo"] = df["p2_status_brn"] + df["p2_status_tox"] + df["p2_status_psn"]

    # p1 - p2
    df["diff_status_stunned"] = df["p1_status_stunned"] - df["p2_status_stunned"]
    df["diff_status_poisoned"] = df["p1_status_poisoned"] - df["p2_status_poisoned"]
    df["diff_status_dmgresiduo"] = df["p1_status_dmgresiduo"] - df["p2_status_dmgresiduo"]

    # survival rate
    df["p1_survival_rate"] = 1 - (df["p1_status_fnt"] / 6)
    df["p2_survival_rate"] = 1 - (df["p2_status_fnt"] / 6)
    df["diff_survival_rate"] = df["p1_survival_rate"] - df["p2_survival_rate"]
    
    return df[["battle_id","diff_status_stunned","diff_status_poisoned","diff_status_dmgresiduo"]].fillna(0).infer_objects()



def ragg_effects(dff,unique):
    p1_effects = effects_counts(dff, 'p1',unique)
    p2_effects = effects_counts(dff, 'p2',unique)
    df = p1_effects.merge(p2_effects, on='battle_id', how='left').fillna(0)
    expected_effects = [
        "clamp", "confusion", "disable", "firespin",
        "reflect", "substitute", "typechange", "wrap"
    ]
    for eff in expected_effects:
        for p in ["p1", "p2"]:
            col = f"{p}_effect_{eff}"
            if col not in df.columns:
                df[col] = 0
    
    # Macro-categories
    df["p1_effect_trap"] = df[["p1_effect_clamp", "p1_effect_firespin", "p1_effect_wrap"]].sum(axis=1)
    df["p1_effect_buff"] = df[["p1_effect_reflect", "p1_effect_substitute"]].sum(axis=1)
    df["p1_effect_confuse"] = df["p1_effect_confusion"]
    df["p1_effect_transform"] = df["p1_effect_typechange"]

    df["p2_effect_trap"] = df[["p2_effect_clamp", "p2_effect_firespin", "p2_effect_wrap"]].sum(axis=1)
    df["p2_effect_buff"] = df[["p2_effect_reflect", "p2_effect_substitute"]].sum(axis=1)
    df["p2_effect_confuse"] = df["p2_effect_confusion"]
    df["p2_effect_transform"] = df["p2_effect_typechange"]

    # p1-p2
    df["diff_effect_trap"] = df["p1_effect_trap"] - df["p2_effect_trap"]
    df["diff_effect_buff"] = df["p1_effect_buff"] - df["p2_effect_buff"]
    df["diff_effect_confuse"] = df["p1_effect_confuse"] - df["p2_effect_confuse"]

    return df[["battle_id","diff_effect_trap","diff_effect_buff","diff_effect_confuse","p1_effect_transform","p2_effect_transform"]].fillna(0).infer_objects()






# Extract features by looking only at data from the p1 squad and the lead pokemon from p2
import pandas as pd

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



#Stats before the match plus looking at battle timeline 
import pandas as pd

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



# Extract features from all the data possible for a single battle
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



# Extract features from all the data possible for a single battle
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