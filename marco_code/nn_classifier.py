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
from sklearn.model_selection import train_test_split
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
    def __init__(self, features, labels=None):
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),   # Direct from input
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 2)  # Output 2 classes
        )
    
    def forward(self, x):
        return self.classifier(x)




# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs):
    
    criterion_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # 'max' because we want to maximize accuracy
        factor=0.5,           # Reduce LR by half
        patience=15,          # Wait 15 epochs without improvement
        verbose=True,         # Print when LR changes
        min_lr=1e-6          # Don't go below this LR
    )    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_class = 0
        train_loss_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - get both reconstruction and classification
            classification = model(X_batch)
            
            # Compute both losses
            loss_class = criterion_class(classification, y_batch)
            
            # Combined loss
            loss = loss_class

            loss.backward()
            optimizer.step()
            train_loss_class += loss.item()
            train_loss_total = train_loss_class
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        val_loss_recon = 0
        val_loss_class = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                classification = model(X_batch)
                
                # Compute validation losses
                val_loss_class += criterion_class(classification, y_batch).item()
                
                # Compute accuracy
                _, predicted = torch.max(classification, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        scheduler.step(acc) 
        # Save best model based on accuracy
        #if acc > best_acc:
        #    best_acc = acc
        #    torch.save(model.state_dict(), './advanced_model.pth')
        
        # Print progress
        train_loss_recon=0
        val_loss_recon = 0
        print(train_loss_class)
        print(train_loss_total)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'  Train - Total: {train_loss_total/len(train_loader):.4f} | '
                  f'Recon: {train_loss_recon/len(train_loader):.4f} | '
                  f'Class: {train_loss_class/len(train_loader):.4f}')
            print(f'  Val   - Recon: {val_loss_recon/len(val_loader):.4f} | '
                  f'Class: {val_loss_class/len(val_loader):.4f} | '
                  f'Acc: {acc:.2f}%')
    
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
    

scaler = StandardScaler()
X_train = np.array([battle_to_full_features(b) for b in train_battles])
y_train = np.array([int(b['player_won']) for b in train_battles])
    
    
X_test = np.array([battle_to_full_features(b) for b in test_battles])
#Test data doesnt have label?
#y_test = np.array([int(b['player_won']) for b in test_battles])
test_ids = [b['battle_id'] for b in test_battles]
    


# Split
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
X_val,X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Normalize using only training statistics
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

   
# Create dataloaders
train_loader = DataLoader(BattleDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(BattleDataset(X_val, y_val), batch_size=16)
test_loader = DataLoader(BattleDataset(X_test, y_test), batch_size=16)
    
# Train
print("Training Advanced Encoder-Decoder with RAW timeline data...")
print("Architecture: Input(1242) → Encoder(256→128→96) → Decoder(32→2)\n")
        
input_dim = X_train.shape[1]
epochs = 400
    
model = SimpleClassifier(input_dim=input_dim).to(device)
model = train_model(model, train_loader, val_loader, epochs)


# Predict
print("\nGenerating predictions...")
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
# Save
submission = pd.DataFrame({
    'battle_id': test_ids,
    'player_won': predictions
})
submission.to_csv('./submission.csv', index=False)
"""
