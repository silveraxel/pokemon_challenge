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