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

# Configuration
MAX_SEQ_LENGTH = 1000  # Reduced from original 10k+ columns for practicality
TIMESTAMP_MEAN = 650
TIMESTAMP_STD = 502

def read_ds_gzip(path: Path = None, ds: str = "TRAIN") -> pd.DataFrame:
    """Read and parse gzipped CSV dataset with proper column handling"""
    with gzip.open(path if path else f'{ds}.CSV.gz') as f:
        lines = [line.decode().strip() for line in f]
        max_columns = max(len(line.split(",")) for line in lines)
        
        file_like = io.BytesIO(b'\n'.join([line.encode() for line in lines]))
        columns = ["battleneturl", "played_race"] if "TRAIN" in ds else ["played_race"]
        columns += list(range(max_columns - len(columns)))
        
        return pd.read_csv(file_like, names=columns, dtype=str)

def process_sequence(row, action_encoder):
    """Process a single row into action indices and normalized timestamps"""
    current_time = 0.0
    actions = []
    timestamps = []
    
    for x in row:
        if pd.isna(x):
            actions.append('no_action')
            timestamps.append((current_time - TIMESTAMP_MEAN) / TIMESTAMP_STD)
        else:
            x_str = str(x)
            if x_str.startswith('t'):
                current_time = float(x_str[1:])
            else:
                actions.append(x_str)
                timestamps.append((current_time - TIMESTAMP_MEAN) / TIMESTAMP_STD)
    
    # Encode actions with padding
    encoded_actions = action_encoder.transform(actions) + 1  # Reserve 0 for padding
    return encoded_actions, np.array(timestamps)

def format_data(df):
    """Process data into sequences with proper timestamp handling"""
    # Encode users and races
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['battleneturl'])
    race_encoder = LabelEncoder()
    df['race_encoded'] = race_encoder.fit_transform(df['played_race'])

    # Build action vocabulary
    action_cols = df.columns[2:-2]
    all_actions = ['no_action']  # Start with padding token
    
    for row in df[action_cols].values:
        for x in row:
            if not pd.isna(x) and not str(x).startswith('t'):
                all_actions.append(str(x))
    
    action_encoder = LabelEncoder()
    action_encoder.fit(np.unique(all_actions))

    # Process sequences
    action_sequences = []
    timestamp_sequences = []
    
    for row in df[action_cols].values:
        encoded, ts = process_sequence(row, action_encoder)
        
        # Pad/truncate sequences
        if len(encoded) > MAX_SEQ_LENGTH:
            action_sequences.append(encoded[:MAX_SEQ_LENGTH])
            timestamp_sequences.append(ts[:MAX_SEQ_LENGTH])
        else:
            pad_len = MAX_SEQ_LENGTH - len(encoded)
            action_sequences.append(np.pad(encoded, (0, pad_len), 'constant'))
            timestamp_sequences.append(np.pad(ts, (0, pad_len), 'constant', constant_values=0))

    return {
        'user_ids': df['user_id'].values,
        'races': df['race_encoded'].values,
        'action_seqs': np.array(action_sequences),
        'time_seqs': np.expand_dims(np.array(timestamp_sequences), -1),
        'encoders': {
            'user': user_encoder,
            'race': race_encoder,
            'action': action_encoder
        }
    }

def build_model(n_users, n_races, n_actions):
    """Enhanced model architecture with proper masking handling"""
    # Sequence inputs
    action_input = layers.Input(shape=(MAX_SEQ_LENGTH,), name='action_input')
    time_input = layers.Input(shape=(MAX_SEQ_LENGTH, 1), name='time_input')
    
    # Action processing with masking
    x = layers.Embedding(
        input_dim=n_actions + 1,  # +1 for padding token
        output_dim=64,
        mask_zero=False  # Turn OFF built-in masking
    )(action_input)

    # Create mask from action_input
    mask = layers.Lambda(
        lambda inp: tf.cast(tf.greater(inp, 0), tf.float32),
        name='action_mask'
    )(action_input)
    mask = layers.Reshape((MAX_SEQ_LENGTH, 1))(mask)

    # Multiply timestamps by our manual mask
    time_masked = layers.Multiply()([time_input, mask])

    # Concatenate embeddings + masked time
    x = layers.Concatenate()([x, time_masked])

    # Then continue with LSTM
    x = layers.Bidirectional(layers.LSTM(128))(x)    
    # Race embedding
    race_input = layers.Input(shape=(1,), name='race_input')
    y = layers.Embedding(n_races, 8)(race_input)
    y = layers.Flatten()(y)
    
    # Combined features
    combined = layers.concatenate([x, y])
    output = layers.Dense(n_users, activation='softmax')(combined)
    
    return Model(inputs=[action_input, time_input, race_input], outputs=output)
# Data processing
df_train = read_ds_gzip(Path('TRAIN.CSV.GZ'))
formatted = format_data(df_train)

# Train/validation split
(X_act_train, X_act_val, 
 X_time_train, X_time_val,
 race_train, race_val,
 y_train, y_val) = train_test_split(
    formatted['action_seqs'],
    formatted['time_seqs'],
    formatted['races'],
    formatted['user_ids'],
    test_size=0.2,
    stratify=formatted['user_ids'],
    random_state=42
)

# Model setup
model = build_model(
    n_users=len(formatted['encoders']['user'].classes_),
    n_races=len(formatted['encoders']['race'].classes_),
    n_actions=len(formatted['encoders']['action'].classes_)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Class weighting
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# Training
history = model.fit(
    x=[X_act_train, X_time_train, race_train],
    y=y_train,
    validation_data=([X_act_val, X_time_val, race_val], y_val),
    epochs=30,
    batch_size=256,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)