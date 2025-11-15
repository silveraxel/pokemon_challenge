#!/usr/bin/env python

from libraries import *

def main():
    # --- Define the path to our data ---
    COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
    DATA_PATH = os.path.join('../input', COMPETITION_NAME)
    train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
    test_file_path = os.path.join(DATA_PATH, 'test.jsonl')

    train_data = []
    test_data  = []

    # --- Load TRAIN data ---
    print(f"📦 Loading data from '{train_file_path}'...")
    try:
        with open(train_file_path, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
        print(f"✅ Successfully loaded {len(train_data)} battles from train.")
    
        # Show structure of first train battle
        if train_data:
            print("\n--- Structure of the first train battle: ---")
            first_battle = train_data[0]
            battle_for_display = first_battle.copy()
            battle_for_display['battle_timeline'] = first_battle.get('battle_timeline', [])[:2]
            print(json.dumps(battle_for_display, indent=4))
            if len(first_battle.get('battle_timeline', [])) > 3:
                print("    ...")
                print("    (battle_timeline has been truncated for display)")

    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the training file at '{train_file_path}'.")
        print("Please make sure you have added the competition data to this notebook.")


    # --- Load TEST data ---
    print(f"\n📦 Loading data from '{test_file_path}'...")
    try:
        with open(test_file_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        print(f"✅ Successfully loaded {len(test_data)} battles from test.")
    
        if test_data:
            print("\n--- Structure of the first test battle: ---")
            first_test_battle = test_data[0]
            test_display = first_test_battle.copy()
            test_display['battle_timeline'] = test_display.get('battle_timeline', [])[:2]
            print(json.dumps(test_display, indent=4))
            if len(first_test_battle.get('battle_timeline', [])) > 3:
                print("    ...")
                print("    (battle_timeline has been truncated for display)")

    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the test file at '{test_file_path}'.")
        print("Please make sure you have added the competition data to this notebook.")

    #train_data e test_data
    #battle,squad,pokemon,timeline
    train_list=create_dataframe(train_data)
    test_list=create_dataframe(test_data)

    # # **TRAINING MODEL**

    # Model of the submission submission_vecchio.csv with public score 0.8373
    # 
    # In the first part of the code, we grouped the features into categories for better understanding of what we were using and also to clarify which variables required scaling.After testing our initial hypothesis, we confirmed that only the numerical features benefited from scaling, while the counter-type features did not. Therefore, we excluded the counter features from the scaling process. Next, we defined a pipeline and applied cross-validation, to prevent overfitting and to evaluate if our features were informative.


    # Loading data
    train_df = extract_all_ragg(train_list,True)
    test_df = extract_all_ragg(test_list,False)

    # Def X e y 
    X = train_df.drop(columns=['player_won', 'battle_id'])
    y = train_df['player_won']

    # Def feature
    type_cols=[ 'p1_dragon', 'p1_electric','p1_fire', 'p1_flying',
            'p1_ghost', 'p1_grass', 'p1_ground', 'p1_ice',
            'p1_normal', 'p1_notype', 'p1_poison', 'p1_psychic', 'p1_rock',
            'p1_water', 'p2_dragon', 'p2_electric','p2_fire', 'p2_flying',
            'p2_ghost', 'p2_grass', 'p2_ground', 'p2_ice','p2_normal',
            'p2_notype', 'p2_poison', 'p2_psychic', 'p2_rock','p2_water']

    status_cols=["diff_status_stunned","diff_status_poisoned","diff_status_dmgresiduo"]

    effect_cols=["diff_effect_trap","diff_effect_buff","diff_effect_confuse",
                "p1_effect_transform","p2_effect_transform"]

    boost_cols= [ 'p1_boost_atk', 'p1_boost_def', 'p1_boost_spa',
        'p1_boost_spd', 'p1_boost_spe', 'p2_boost_atk', 'p2_boost_def',
        'p2_boost_spa', 'p2_boost_spd', 'p2_boost_spe',]

    move_type_cols=[ 'p1_movetype_ELECTRIC',
        'p1_movetype_FIGHTING', 'p1_movetype_FIRE', 'p1_movetype_FLYING',
        'p1_movetype_GHOST', 'p1_movetype_GRASS', 'p1_movetype_GROUND',
        'p1_movetype_ICE', 'p1_movetype_NORMAL', 'p1_movetype_POISON',
        'p1_movetype_PSYCHIC', 'p1_movetype_ROCK', 'p1_movetype_WATER',
        'p2_movetype_ELECTRIC', 'p2_movetype_FIGHTING', 'p2_movetype_FIRE',
        'p2_movetype_FLYING', 'p2_movetype_GHOST', 'p2_movetype_GRASS',
        'p2_movetype_GROUND', 'p2_movetype_ICE', 'p2_movetype_NORMAL',
        'p2_movetype_POISON', 'p2_movetype_PSYCHIC', 'p2_movetype_ROCK',
        'p2_movetype_WATER']

    move_cat_cols=['p1_movecat_PHYSICAL', 'p1_movecat_SPECIAL', 'p1_movecat_STATUS',
            'p2_movecat_PHYSICAL', 'p2_movecat_SPECIAL','p2_movecat_STATUS']

    match_cols= ['hp_diff_mean','diff_base_hp','diff_base_atk', 'diff_base_def', 
                'diff_base_spa', 'diff_base_spd', 'diff_base_spe', 'diff_level',
                'move_basepow_diff_mean','move_acc_diff_mean', 'somma_diff_hp']

    # Features to scale
    temp = type_cols + status_cols + effect_cols + move_type_cols + move_cat_cols
    exclude = ['battle_id','player_won'] + temp
    features = [col for col in train_df.columns if col not in exclude]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), features),
        ('pass_types', 'passthrough', temp)
    ], remainder='drop')

    # Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=5000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Prediction out-of-fold
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method='predict')
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    print("=== Results of Cross-Validation (10-fold) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred, digits=3))

    # Final fit on all data
    pipe.fit(X, y)

    X_test = test_df.drop(columns=['battle_id'])
    test_predictions = pipe.predict(X_test)
    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })
    submission_df.to_csv('submission_vecchio.csv', index=False)
    print("'submission_vecchio.csv' file created successfully!")
    display(submission_df.head())
    display(submission_df.nunique(axis=0))


    # To get to our best model, we applied a gridsearch to find the best hyperparameter for the LogisticRegression of our previous model.


    # Loading data
    train_df = extract_all_ragg(train_list, True)
    X = train_df.drop(columns=['player_won', 'battle_id'])
    y = train_df['player_won']

    # Def feature
    type_cols=[ 'p1_dragon', 'p1_electric','p1_fire', 'p1_flying',
            'p1_ghost', 'p1_grass', 'p1_ground', 'p1_ice',
            'p1_normal', 'p1_notype', 'p1_poison', 'p1_psychic', 'p1_rock',
            'p1_water', 'p2_dragon', 'p2_electric','p2_fire', 'p2_flying',
            'p2_ghost', 'p2_grass', 'p2_ground', 'p2_ice','p2_normal',
            'p2_notype', 'p2_poison', 'p2_psychic', 'p2_rock','p2_water']

    status_cols=["diff_status_stunned","diff_status_poisoned","diff_status_dmgresiduo"]

    effect_cols=["diff_effect_trap","diff_effect_buff","diff_effect_confuse",
                "p1_effect_transform","p2_effect_transform"]

    boost_cols= [ 'p1_boost_atk', 'p1_boost_def', 'p1_boost_spa',
        'p1_boost_spd', 'p1_boost_spe', 'p2_boost_atk', 'p2_boost_def',
        'p2_boost_spa', 'p2_boost_spd', 'p2_boost_spe',]

    move_type_cols=[ 'p1_movetype_ELECTRIC',
        'p1_movetype_FIGHTING', 'p1_movetype_FIRE', 'p1_movetype_FLYING',
        'p1_movetype_GHOST', 'p1_movetype_GRASS', 'p1_movetype_GROUND',
        'p1_movetype_ICE', 'p1_movetype_NORMAL', 'p1_movetype_POISON',
        'p1_movetype_PSYCHIC', 'p1_movetype_ROCK', 'p1_movetype_WATER',
        'p2_movetype_ELECTRIC', 'p2_movetype_FIGHTING', 'p2_movetype_FIRE',
        'p2_movetype_FLYING', 'p2_movetype_GHOST', 'p2_movetype_GRASS',
        'p2_movetype_GROUND', 'p2_movetype_ICE', 'p2_movetype_NORMAL',
        'p2_movetype_POISON', 'p2_movetype_PSYCHIC', 'p2_movetype_ROCK',
        'p2_movetype_WATER']

    move_cat_cols=['p1_movecat_PHYSICAL', 'p1_movecat_SPECIAL', 'p1_movecat_STATUS',
            'p2_movecat_PHYSICAL', 'p2_movecat_SPECIAL','p2_movecat_STATUS']

    match_cols= ['hp_diff_mean','diff_base_hp','diff_base_atk', 'diff_base_def', 
                'diff_base_spa', 'diff_base_spd', 'diff_base_spe', 'diff_level',
                'move_basepow_diff_mean','move_acc_diff_mean', 'somma_diff_hp']

    # Features to scale
    temp = type_cols + status_cols + effect_cols + move_type_cols + move_cat_cols
    exclude = ['battle_id','player_won'] + temp
    features = [col for col in train_df.columns if col not in exclude]
    print("Features da scalare:", len(features))

    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), features),
        ('pass_types', 'passthrough', temp)
    ], remainder='drop')

    # Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=5000, random_state=42))
    ])

    # Parameters to checks 
    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'lbfgs'],  # liblinear supporta l1, lbfgs no
        'model__class_weight': [None, 'balanced']
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True, 
        refit=True
    )

    grid_search.fit(X, y)
    print("Best parameters:")
    print(grid_search.best_params_)

    print(f" best accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred = cross_val_predict(best_model, X, y, cv=cv, method='predict')
    y_prob = cross_val_predict(best_model, X, y, cv=cv, method='predict_proba')[:, 1]

    print("final result")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall   : {recall_score(y, y_pred):.4f}")
    print(f"F1-score : {f1_score(y, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y, y_prob):.4f}")

    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred, digits=3))


    # Model of the submission submission_cc.csv with public score 0.8380. We trained directly the model with the best hyperparameter. In the first part we have also introduced drop_cols, as the name says features to drop given collinearity and feature importance.


    # Loading data
    train_df = extract_all_ragg(train_list, True)
    test_df  = extract_all_ragg(test_list, False)

    # Remove the collinear columns (Given the result of the next analysis)
    drop_cols = [
        "p2_ground", "diff_base_spd", "p2_effect_transform",
        "p2_boost_spa", "p2_boost_spd", "p1_ground"
    ]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df  = test_df.drop(columns=drop_cols, errors="ignore")

    X = train_df.drop(columns=["player_won", "battle_id"], errors="ignore")
    y = train_df["player_won"]

    # Cleaner way for group of features
    all_cols = X.columns.tolist()

    type_cols = [c for c in all_cols if ("p1_" in c or "p2_" in c) and any(t in c for t in [
        "dragon","electric","fire","flying","ghost","grass","ground","ice",
        "normal","notype","poison","psychic","rock","water"
    ])]
    status_cols = [c for c in all_cols if "diff_status" in c]
    effect_cols = [c for c in all_cols if "effect_" in c]
    boost_cols  = [c for c in all_cols if "boost_" in c]
    move_type_cols = [c for c in all_cols if "movetype" in c]
    move_cat_cols  = [c for c in all_cols if "movecat" in c]
    match_cols = [c for c in all_cols if any(k in c for k in [
        "hp_diff", "diff_base", "diff_level", "move_basepow", "move_acc", "somma_diff"
    ])]

    temp = type_cols + status_cols + effect_cols + move_type_cols + move_cat_cols
    exclude = ["battle_id", "player_won"] + temp
    features = [c for c in all_cols if c not in exclude]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), features),
        ("pass_types", "passthrough", temp)
    ], remainder="drop")

    # Pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42,
            C=0.5,
            penalty="l1",
            solver="liblinear"
        ))
    ])


    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    # Metrics
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)
    roc  = roc_auc_score(y, y_prob)

    print("Cross-validation results(10-fold)")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred, digits=3))

    # Final fit
    pipe.fit(X, y)
    X_test = test_df.drop(columns=["battle_id"], errors="ignore")
    test_predictions = pipe.predict(X_test)
    submission_df = pd.DataFrame({
        "battle_id": test_df["battle_id"],
        "player_won": test_predictions
    })
    submission_df.to_csv("submission.csv", index=False)

    print("File 'submission_cc.csv' file created successfully!")
    display(submission_df.head())
    display(submission_df.nunique(axis=0))


    # Model of the submission submission_totti.csv with public score 0.8366. Here our idea was to reduce the gap of knowledge between the 2 teams using extract_all_tottipi() and the best hyperparameter but probably we added more noise than expected

    # Load data
    train_df = extract_all_tottipi(train_list,True)
    test_df = extract_all_tottipi(test_list,False) 

    # Def X, y
    X = train_df.drop(columns=['player_won', 'battle_id'])
    y = train_df['player_won']

    # Feature groups
    type_cols=[ 'p1_dragon', 'p1_electric','p1_fire', 'p1_flying',
            'p1_ghost', 'p1_grass', 'p1_ground', 'p1_ice',
            'p1_normal', 'p1_notype', 'p1_poison', 'p1_psychic', 'p1_rock',
            'p1_water', 'p2_dragon', 'p2_electric','p2_fire', 'p2_flying',
            'p2_ghost', 'p2_grass', 'p2_ground', 'p2_ice','p2_normal',
            'p2_notype', 'p2_poison', 'p2_psychic', 'p2_rock','p2_water']

    status_cols=["diff_status_stunned","diff_status_poisoned","diff_status_dmgresiduo"]

    effect_cols=["diff_effect_trap","diff_effect_buff","diff_effect_confuse",
                "p1_effect_transform","p2_effect_transform"]

    boost_cols= [ 'p1_boost_atk', 'p1_boost_def', 'p1_boost_spa',
        'p1_boost_spd', 'p1_boost_spe', 'p2_boost_atk', 'p2_boost_def',
        'p2_boost_spa', 'p2_boost_spd', 'p2_boost_spe',]

    move_type_cols=[ 'p1_movetype_ELECTRIC',
        'p1_movetype_FIGHTING', 'p1_movetype_FIRE', 'p1_movetype_FLYING',
        'p1_movetype_GHOST', 'p1_movetype_GRASS', 'p1_movetype_GROUND',
        'p1_movetype_ICE', 'p1_movetype_NORMAL', 'p1_movetype_POISON',
        'p1_movetype_PSYCHIC', 'p1_movetype_ROCK', 'p1_movetype_WATER',
        'p2_movetype_ELECTRIC', 'p2_movetype_FIGHTING', 'p2_movetype_FIRE',
        'p2_movetype_FLYING', 'p2_movetype_GHOST', 'p2_movetype_GRASS',
        'p2_movetype_GROUND', 'p2_movetype_ICE', 'p2_movetype_NORMAL',
        'p2_movetype_POISON', 'p2_movetype_PSYCHIC', 'p2_movetype_ROCK',
        'p2_movetype_WATER']

    move_cat_cols=['p1_movecat_PHYSICAL', 'p1_movecat_SPECIAL', 'p1_movecat_STATUS',
            'p2_movecat_PHYSICAL', 'p2_movecat_SPECIAL','p2_movecat_STATUS']

    match_cols= ['hp_diff_mean','diff_base_hp','diff_base_atk', 'diff_base_def', 
                'diff_base_spa', 'diff_base_spd', 'diff_base_spe', 'diff_level',
                'move_basepow_diff_mean','move_acc_diff_mean', 'somma_diff_hp']

    # Features to scale
    temp = type_cols + status_cols + effect_cols + move_type_cols + move_cat_cols
    exclude = ['battle_id','player_won'] + temp
    features = [col for col in train_df.columns if col not in exclude]

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), features),
        ('pass_types', 'passthrough', temp)
    ], remainder='drop')

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            max_iter=5000, 
            random_state=42,
            C=0.5,#0.7,#  
            penalty='l1',#'l2',#
            solver='liblinear'
            ))#C=0.7, penalty='l2'
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    y_pred = cross_val_predict(pipe, X, y, cv=cv, method='predict')
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    print("=== Results of Cross-Validation (10-fold) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred, digits=3))

    # Final fit
    pipe.fit(X, y)
    X_test = test_df.drop(columns=['battle_id'])
    test_predictions = pipe.predict(X_test)

    submission_df = pd.DataFrame({
        'battle_id': test_df['battle_id'],
        'player_won': test_predictions
    })

    submission_df.to_csv('submission_totti.csv', index=False)

    print("\n✅ 'submission_totti.csv' file created successfully!")
    display(submission_df.head())
    display(submission_df.nunique(axis=0))



if __name__ == '__main__':
    main()
