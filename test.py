import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import gzip
from pathlib import Path
import os
from typing import Optional

def read_ds_gzip(path: Optional[Path]=None, ds: str = "TRAIN") -> pd.DataFrame:
    """Read and parse the gzipped CSV dataset.
    
    Args:
        path (Optional[Path]): Path to the dataset file
        ds (str): Dataset type ("TRAIN" or "TEST"). Defaults to "TRAIN"
    
    Returns:
        pd.DataFrame: Parsed dataset
    """
    with gzip.open(path if path else f'{ds}.CSV.gz') as f:
        max_actions = max(len(str(c).split(",")) for c in f.readlines())
        f.seek(0)
        _names = ["battleneturl", "played_race"] if "TRAIN" in ds else ["played_race"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype=str)

# First create and fit the action encoder globally
action_encoder = LabelEncoder()

# Load training data
df_train = read_ds_gzip(Path('TRAIN.CSV.GZ'))

def format_data(df):
    # 1. Encode Target
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['battleneturl'])
    
    # 2. Process Race
    race_encoder = LabelEncoder()
    df['race_encoded'] = race_encoder.fit_transform(df['played_race'])
    
    # 3. Process Action Sequence
    action_columns = df.columns[2:-2]  # Exclude URL, user_id, race_encoded
    
    # Build action vocabulary first
    all_actions = []
    for _, row in df.iterrows():
        for col in action_columns:
            action = row[col]
            if pd.isna(action):
                all_actions.append('no_action')
            elif action.startswith('t'):
                all_actions.append('time_marker')
            else:
                all_actions.append(action)
    
    # Fit encoder on complete vocabulary
    action_encoder.fit(np.unique(all_actions))
    
    # Now process sequences with fitted encoder
    sequences = []
    for _, row in df.iterrows():
        seq = []
        for col in action_columns:
            action = row[col]
            if pd.isna(action):
                action = 'no_action'
            elif action.startswith('t'):
                action = 'time_marker'
            seq.append(action)
        sequences.append(action_encoder.transform(seq))
    
    return {
        'user_ids': df['user_id'],
        'races': df['race_encoded'],
        'sequences': np.array(sequences),
        'user_encoder': user_encoder,
        'action_encoder': action_encoder,
        'race_encoder': race_encoder
    }

def build_model(n_users, n_races):
    # Input 1: Action sequences
    action_input = layers.Input(shape=(None,), name='action_input')
    action_emb = layers.Embedding(
        input_dim=len(action_encoder.classes_),  # Use encoder's learned classes
        output_dim=64,
        mask_zero=True
    )(action_input)
    
    # Input 2: Race
    race_input = layers.Input(shape=(1,), name='race_input')
    race_emb = layers.Embedding(n_races, 8)(race_input)
    race_flat = layers.Flatten()(race_emb)
    
    # Sequence processing
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(action_emb)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Combine features
    combined = layers.concatenate([x, race_flat])
    
    # Output
    output = layers.Dense(n_users, activation='softmax')(combined)
    
    return Model(
        inputs=[action_input, race_input],
        outputs=output
    )



# Usage
formatted = format_data(df_train)

model = build_model(
    n_users=len(formatted['user_encoder'].classes_),
    n_races=len(formatted['race_encoder'].classes_)
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x=[formatted['sequences'], formatted['races']],
    y=formatted['user_ids'],
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
