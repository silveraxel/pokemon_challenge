#!/usr/bin/env python
# coding: utf-8

"""
ADVANCED: Encoder-Decoder with Dynamic Battle Timeline Features

This version processes the battle_timeline to extract dynamic features:
- Number of turns
- Move types used
- Damage dealt
- Switches made
- Status conditions applied
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DYNAMIC FEATURE EXTRACTION
# ============================================================================

def extract_timeline_features(battle_timeline, max_turns=30):
    """
    Extract RAW turn-by-turn information from battle timeline
    
    For each turn, extract:
    - P1 pokemon state: hp_pct, status (encoded), boosts (5 stats)
    - P1 move: type (encoded), category (encoded), base_power, accuracy, priority
    - P2 pokemon state: hp_pct, status (encoded), boosts (5 stats)
    - P2 move: type (encoded), category (encoded), base_power, accuracy, priority
    
    Total per turn: 24 features
    - P1 state: 1 (hp) + 1 (status) + 5 (boosts) = 7
    - P1 move: 1 (type) + 1 (category) + 1 (power) + 1 (accuracy) + 1 (priority) = 5
    - P2 state: 1 (hp) + 1 (status) + 5 (boosts) = 7
    - P2 move: 1 (type) + 1 (category) + 1 (power) + 1 (accuracy) + 1 (priority) = 5
    
    Total features: 24 * max_turns (default: 24 * 50 = 1200)
    
    Pad or truncate to max_turns.
    """
    
    # Status encoding
    status_map = {
        'nostatus': 0, 'par': 1, 'slp': 2, 'frz': 3, 
        'psn': 4, 'tox': 5, 'brn': 6, 'fnt': 7
    }
    
    # Move type encoding (common Pokemon types)
    type_map = {
        'NORMAL': 0, 'FIRE': 1, 'WATER': 2, 'ELECTRIC': 3, 'GRASS': 4,
        'ICE': 5, 'FIGHTING': 6, 'POISON': 7, 'GROUND': 8, 'FLYING': 9,
        'PSYCHIC': 10, 'BUG': 11, 'ROCK': 12, 'GHOST': 13, 'DRAGON': 14,
        'DARK': 15, 'STEEL': 16, 'FAIRY': 17
    }
    
    # Move category encoding
    category_map = {'PHYSICAL': 0, 'SPECIAL': 1, 'STATUS': 2}
    
    all_features = []
    
    for turn_idx in range(max_turns):
        if turn_idx < len(battle_timeline):
            turn = battle_timeline[turn_idx]
            # --- P1 Pokemon State (7 features) ---
            p1_state = turn.get('p1_pokemon_state', {})
            p1_hp = p1_state.get('hp_pct', 0.0)
            p1_status = status_map.get(p1_state.get('status', 'nostatus'), 0)
            p1_boosts = p1_state.get('boosts', {})
            p1_boost_atk = p1_boosts.get('atk', 0)
            p1_boost_def = p1_boosts.get('def', 0)
            p1_boost_spa = p1_boosts.get('spa', 0)
            p1_boost_spd = p1_boosts.get('spd', 0)
            p1_boost_spe = p1_boosts.get('spe', 0)
            
            # --- P1 Move Details (5 features) ---
            p1_move = turn.get('p1_move_details')
            if p1_move is not None:
                p1_move_type = type_map.get(p1_move.get('type'), 0)
                p1_move_category = category_map.get(p1_move.get('category'), 0)
                p1_move_power = p1_move.get('base_power', 0) / 200.0  # Normalize to 0-1
                p1_move_accuracy = p1_move.get('accuracy', 1.0)
                p1_move_priority = (p1_move.get('priority', 0) + 5) / 10.0  # Normalize -5 to +5 → 0 to 1
            else:
                p1_move_type = -1  # No move indicator
                p1_move_category = -1
                p1_move_power = 0.0
                p1_move_accuracy = 0.0
                p1_move_priority = 0.5
            
            # --- P2 Pokemon State (7 features) ---
            p2_state = turn.get('p2_pokemon_state', {})
            p2_hp = p2_state.get('hp_pct', 0.0)
            p2_status = status_map.get(p2_state.get('status', 'nostatus'), 0)
            p2_boosts = p2_state.get('boosts', {})
            p2_boost_atk = p2_boosts.get('atk', 0)
            p2_boost_def = p2_boosts.get('def', 0)
            p2_boost_spa = p2_boosts.get('spa', 0)
            p2_boost_spd = p2_boosts.get('spd', 0)
            p2_boost_spe = p2_boosts.get('spe', 0)
            
            # --- P2 Move Details (5 features) ---
            p2_move = turn.get('p2_move_details')
            if p2_move is not None:
                p2_move_type = type_map.get(p2_move.get('type', 'NORMAL'), 0)
                p2_move_category = category_map.get(p2_move.get('category', 'PHYSICAL'), 0)
                p2_move_power = p2_move.get('base_power', 0) / 200.0  # Normalize to 0-1
                p2_move_accuracy = p2_move.get('accuracy', 1.0)
                p2_move_priority = (p2_move.get('priority', 0) + 5) / 10.0  # Normalize
            else:
                p2_move_type = -1  # No move indicator
                p2_move_category = -1
                p2_move_power = 0.0
                p2_move_accuracy = 0.0
                p2_move_priority = 0.5
            
            # Combine all features for this turn (24 features)
            turn_features = [
                # P1 state (7)
                p1_hp, float(p1_status), float(p1_boost_atk), float(p1_boost_def), 
                float(p1_boost_spa), float(p1_boost_spd), float(p1_boost_spe),
                # P1 move (5)
                float(p1_move_type), float(p1_move_category), p1_move_power, 
                p1_move_accuracy, p1_move_priority,
                # P2 state (7)
                p2_hp, float(p2_status), float(p2_boost_atk), float(p2_boost_def),
                float(p2_boost_spa), float(p2_boost_spd), float(p2_boost_spe),
                # P2 move (5)
                float(p2_move_type), float(p2_move_category), p2_move_power,
                p2_move_accuracy, p2_move_priority
            ]
        else:
            # Padding with zeros for missing turns
            turn_features = [0.0] * 24

        all_features.extend(turn_features)
    
    return all_features



def extract_static_features(battle):
    """Extract static features (team stats) - same as before"""
    features = []
    
    # Player 1 team (36 features)
    p1_team = battle.get('p1_team_details', [])
    for i in range(6):
        if i < len(p1_team):
            pokemon = p1_team[i]
            features.extend([
                pokemon.get('base_hp', 0),
                pokemon.get('base_atk', 0),
                pokemon.get('base_def', 0),
                pokemon.get('base_spa', 0),
                pokemon.get('base_spd', 0),
                pokemon.get('base_spe', 0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
    
    # Player 2 lead (6 features)
    p2_lead = battle.get('p2_lead_details', {})
    features.extend([
        p2_lead.get('base_hp', 0),
        p2_lead.get('base_atk', 0),
        p2_lead.get('base_def', 0),
        p2_lead.get('base_spa', 0),
        p2_lead.get('base_spd', 0),
        p2_lead.get('base_spe', 0)
    ])
    
    return features


def battle_to_full_features(battle, max_turns=30):
    """
    Combine static AND raw dynamic features
    
    Static: 42 features (team stats)
    Dynamic: 24 * max_turns features (raw turn-by-turn data)
    Total: 42 + 1200 = 1242 features (if max_turns=50)
    
    Each turn contributes 24 features:
    - P1 state: HP%, status, boosts (7 features)
    - P1 move: type, category, power, accuracy, priority (5 features)
    - P2 state: HP%, status, boosts (7 features)
    - P2 move: type, category, power, accuracy, priority (5 features)
    """
    # Get static features (42)
    static = extract_static_features(battle)
    
    # Get raw dynamic features from timeline (24 * max_turns)
    timeline = battle.get('battle_timeline', [])
    dynamic = extract_timeline_features(timeline, max_turns=max_turns)
    
    # Combine: 42 static + (24 * max_turns) dynamic
    return np.array(static + dynamic, dtype=np.float32)


# ============================================================================
# DATASET AND MODEL 
# ============================================================================

class BattleDataset(Dataset):
    def __init__(self, features, labels=None,training=False, noise_std=0.001):
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(labels) if labels is not None else None
        self.training = training
        self.noise_std = noise_std
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Add noise during training only
        x = self.X[idx]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * (torch.abs(x) * self.noise_std + 1e-6)  # Add small epsilon
            x = x + noise
        if self.y is not None:
            return x, self.y[idx]
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),   # Direct from input
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 2)  # Output 2 classes
        )
    
    def forward(self, x):
        return self.classifier(x)




# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs):
    
    criterion_class = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # 'max' because we want to maximize accuracy
        factor=0.5,           # Reduce LR by half
        patience=15,          # Wait 15 epochs without improvement
        verbose=True,         # Print when LR changes
        min_lr=1e-6          # Don't go below this LR
    )    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #best_acc = 0
    try:
        model.load_state_dict(torch.load('./advanced_model.pth'))
    except:
        print('Not available model to be loaded.')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_class = 0
        train_loss_total = 0
        total = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get both reconstruction and classification
            classification = model(X_batch)

            # Compute both losses
            loss_class = criterion_class(classification, y_batch)

            _, predicted = torch.max(classification, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            # Combined loss
            loss = loss_class

            loss.backward()
            optimizer.step()
            train_loss_class += loss.item()
        train__acc = 100 * correct / total
        # Validate
        model.eval()
        correct = 0
        total = 0
        val_loss_class = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                classification = model(X_batch)
                
                # Compute validation losses
                val_loss_class += criterion_class(classification, y_batch).item()
                
                # Compute validation accuracy
                _, predicted = torch.max(classification, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step(val_acc) 
        # Save best model based on accuracy
        #if val_acc > best_acc:
        #    best_acc = val_acc
        #    torch.save(model.state_dict(), './advanced_model.pth')
        

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'  Train - 'f'Class: {train_loss_class/len(train_loader):.4f} | '
                  f'Train Acc: {train__acc:.2f}%')
            print(f'  Val   - 'f'Class: {val_loss_class/len(val_loader):.4f} | '
                  f'Val Acc: {val_acc:.2f}%')
    
    torch.save(model.state_dict(), './advanced_model.pth')
    
    # Load best model
    #model.load_state_dict(torch.load('./advanced_model.pth'))
    #print(f'\nBest Validation Accuracy: {best_acc:.2f}%')
    return model


# ============================================================================
# MAIN
# ============================================================================



COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('../input', COMPETITION_NAME)
    
# Load data
print("Loading data...")
with open(os.path.join(DATA_PATH, 'train.jsonl'), 'r') as f:
    train_battles = [json.loads(line) for line in f]
    
with open(os.path.join(DATA_PATH, 'test.jsonl'), 'r') as f:
    test_battles = [json.loads(line) for line in f]
    
print(f"Train: {len(train_battles)} | Test: {len(test_battles)}")
    

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler() 
X = np.array([battle_to_full_features(b) for b in train_battles])
y = np.array([int(b['player_won']) for b in train_battles])
input_dim = X.shape[1]
epochs = 100
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print('Starting training on new fold')
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    X_train_fold = scaler.fit_transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)
   
# Create dataloaders
    train_loader = DataLoader(BattleDataset(X_train_fold, y_train_fold,training=True, noise_std=0.1), batch_size=32, shuffle=True)
    val_loader = DataLoader(BattleDataset(X_val_fold, y_val_fold,training=False), batch_size=32)

    model = SimpleClassifier(input_dim=input_dim).to(device)
    model = train_model(model, train_loader, val_loader, epochs)
#After the k-fold cross validation save the model
# Predict
""""
print("\nGenerating predictions for testing...")
model.eval()

predictions = []
accuracy = 0
i=0
with torch.no_grad():
    for test_batch in test_loader:
        if isinstance(test_batch, tuple):
            X_batch = test_batch[0]
            y_batch = test_batch[1]
        X_batch = test_batch[0].to(device)
        y_batch = test_batch[1].to(device)
        outputs = model.forward(X_batch)
        preds = torch.max(outputs, 1)
        correct = torch.sum(preds[1] == y_batch).item()
        accuracy += 100 * correct / len(y_batch)
        predictions.extend(preds[1].cpu().numpy())
        i += 1

print('mean testing accuracy = ' + str(accuracy/i))
"""

X_competition = np.array([battle_to_full_features(b) for b in test_battles])
test_ids = [b['battle_id'] for b in test_battles]
X_competition = scaler.transform(X_competition)
competition_loader = DataLoader(BattleDataset(X_competition,training=False),batch_size=32)

# Predict for the competition
print("\nGenerating predictions for the competition...")
model.eval()
predictions = []
with torch.no_grad():
    for test_batch in competition_loader:
        outputs = model.forward(test_batch.to(device))
        preds = torch.max(outputs, 1)
        predictions.extend(preds[1].cpu().numpy())


# Save
submission = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': predictions
})
submission.to_csv('./submission.csv', index=False)
