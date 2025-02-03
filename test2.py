import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import gzip
from pathlib import Path
import os
import io

def read_ds_gzip(path: Path = None, ds: str = "TRAIN") -> pd.DataFrame:
    """Read and parse the gzipped CSV dataset"""
    with gzip.open(path if path else f'{ds}.CSV.gz') as f:
        lines = [line.decode().strip() for line in f]
        max_columns = max(len(line.split(",")) for line in lines)
        
        file_like = io.BytesIO(b'\n'.join([line.encode() for line in lines]))
        columns = ["battleneturl", "played_race"] if "TRAIN" in ds else ["played_race"]
        columns += list(range(max_columns - len(columns)))
        
        return pd.read_csv(file_like, names=columns, dtype=str)

def format_data(df):
    """Process data and create sequences with padding"""
    # Encode users
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['battleneturl'])
    
    # Encode races
    race_encoder = LabelEncoder()
    df['race_encoded'] = race_encoder.fit_transform(df['played_race'])
    
    # Process action sequences
    action_cols = df.columns[2:-2]
    max_length = len(action_cols)
    
    # Build action vocabulary with proper type handling
    all_actions = []
    for row in df[action_cols].values:
        valid_actions = []
        for x in row:
            if pd.isna(x):
                valid_actions.append('no_action')
            else:
                # Convert to string first to handle any numeric values
                x_str = str(x)
                if x_str.startswith('t'):
                    valid_actions.append('time_marker')
                else:
                    valid_actions.append(x_str)
        all_actions.extend(valid_actions)
    
    # Create encoder with numpy for unique values
    action_encoder = LabelEncoder()
    unique_actions = np.unique(all_actions)  # Fixes the warning
    unique_actions = sorted(unique_actions, key=lambda x: x != 'no_action')
    action_encoder.fit(unique_actions)
    
    # Create padded sequences
    sequences = []
    for row in df[action_cols].values:
        processed = []
        for x in row:
            if pd.isna(x):
                processed.append('no_action')
            else:
                x_str = str(x)
                processed.append('time_marker' if x_str.startswith('t') else x_str)
        sequences.append(action_encoder.transform(processed))
    
    return {
        'user_ids': df['user_id'].values,
        'races': df['race_encoded'].values,
        'sequences': np.array(sequences),
        'encoders': {
            'user': user_encoder,
            'race': race_encoder,
            'action': action_encoder
        }
    }
def build_model(n_users, n_races, n_actions):
    """Simplified model architecture"""
    # Action sequence processing
    action_input = layers.Input(shape=(None,), name='action_input')
    x = layers.Embedding(n_actions, 64, mask_zero=True)(action_input)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    # Race feature processing
    race_input = layers.Input(shape=(1,), name='race_input')
    y = layers.Embedding(n_races, 8)(race_input)
    y = layers.Flatten()(y)
    
    # Combined model
    combined = layers.concatenate([x, y])
    output = layers.Dense(n_users, activation='softmax')(combined)
    
    return Model(inputs=[action_input, race_input], outputs=output)

# Load and process data
df_train = read_ds_gzip(Path('TRAIN.CSV.GZ'))
formatted = format_data(df_train)

# Split data
X_train, X_val, race_train, race_val, y_train, y_val = train_test_split(
    formatted['sequences'],
    formatted['races'],
    formatted['user_ids'],
    test_size=0.2,
    stratify=formatted['user_ids'],
    random_state=42
)

# Build and compile model
model = build_model(
    n_users=len(formatted['encoders']['user'].classes_),
    n_races=len(formatted['encoders']['race'].classes_),
    n_actions=len(formatted['encoders']['action'].classes_)
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x=[X_train, race_train],
    y=y_train,
    validation_data=([X_val, race_val], y_val),
    epochs=50,
    batch_size=128,
    class_weight=dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    ))
)