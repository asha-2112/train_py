import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.cluster import KMeans
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data', 'marine_engine_fault_dataset.csv')
OUT = os.path.join(BASE, 'outputs')
ANALYSIS = os.path.join(BASE, 'analysis_outputs')
MODEL_DIR = os.path.join(BASE, 'model')

SENSORS = [
    'Shaft_RPM','Engine_Load','Fuel_Flow','Air_Pressure','Ambient_Temp',
    'Oil_Temp','Oil_Pressure','Vibration_X','Vibration_Y','Vibration_Z',
    'Cylinder1_Pressure','Cylinder1_Exhaust_Temp','Cylinder2_Pressure',
    'Cylinder2_Exhaust_Temp','Cylinder3_Pressure','Cylinder3_Exhaust_Temp',
    'Cylinder4_Pressure','Cylinder4_Exhaust_Temp'
]
LABEL_NAMES = ['Normal'] + [f'Fault {i}' for i in range(1, 8)]
SEQ_LENGTHS = [10, 20, 30]
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def reset_generated_dirs():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ANALYSIS, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for folder in (OUT, ANALYSIS, MODEL_DIR):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def make_sequences(X, y, ts, seq):
    xs, ys, t = [], [], []
    for i in range(seq - 1, len(X)):
        xs.append(X[i-seq+1:i+1])
        ys.append(y[i])
        t.append(ts[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int), np.asarray(t)


def build_model(seq_len):
    model = Sequential([
        Input(shape=(seq_len, len(SENSORS))),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu', name='embedding'),
        Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_one(seq_len, Xs, y, ts, tr_end, va_end):
    Xtr, ytr, _ = make_sequences(Xs[:tr_end], y[:tr_end], ts[:tr_end], seq_len)
    Xva, yva, _ = make_sequences(Xs[tr_end:va_end], y[tr_end:va_end], ts[tr_end:va_end], seq_len)
    Xte, yte, tte = make_sequences(Xs[va_end:], y[va_end:], ts[va_end:], seq_len)

    model = build_model(seq_len)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=40,
        batch_size=64,
        verbose=2,
        callbacks=callbacks
    )

    val_prob = model.predict(Xva, verbose=0)
    val_pred = val_prob.argmax(axis=1)
    _, _, val_f1, _ = precision_recall_fscore_support(
        yva, val_pred, labels=list(range(8)), average='macro', zero_division=0
    )

    test_prob = model.predict(Xte, verbose=0)
    test_pred = test_prob.argmax(axis=1)
    test_acc = accuracy_score(yte, test_pred)
    _, _, test_f1, _ = precision_recall_fscore_support(
        yte, test_pred, labels=list(range(8)), average='macro', zero_division=0
    )

    return model, history, val_f1, test_acc, test_f1, Xte, yte, tte, test_prob


def sensor_profile(fault_df, normal_df, top_n=5):
    rows = []
    for sensor in SENSORS:
        normal_median = float(normal_df[sensor].median())
        fault_median = float(fault_df[sensor].median())
        pct = ((fault_median - normal_median) / normal_median * 100) if normal_median != 0 else np.nan
        rows.append((sensor, normal_median, fault_median, pct))
    rows.sort(key=lambda x: abs(x[3]) if np.isfinite(x[3]) else -1, reverse=True)
    return rows[:top_n]



def generate_fault_class_reference(patterns, contrib_df, selected_seq, acc, mp, mr, mf):
    """Generate one clean CSV reference for all fault-class information."""
    rows = []

    # Fault class summary
    class_info = {
        0: 'Baseline / normal reference',
        1: '2 behavioural patterns identified',
        2: '3 behavioural patterns identified',
        3: '2 behavioural patterns identified',
        4: 'Contributing parameters analysed',
        5: 'Contributing parameters analysed',
        6: 'Contributing parameters analysed',
        7: 'Contributing parameters analysed',
    }
    for fault in range(8):
        rows.append({
            'Section': 'Fault Class Summary', 'Fault': f'Fault {fault}',
            'Pattern_or_Rank': '', 'Samples': '', 'Parameter': '',
            'Observed_Behaviour_or_Change': '', 'Interpretation': class_info[fault]
        })

    # Behavioural patterns for Faults 1-3
    if patterns is not None and not patterns.empty:
        for _, r in patterns.iterrows():
            rows.append({
                'Section': 'Behavioural Pattern',
                'Fault': f"Fault {int(r['Fault'])}",
                'Pattern_or_Rank': f"Pattern {int(r['Pattern'])}",
                'Samples': int(r['Samples']),
                'Parameter': str(r['Top_Parameters']),
                'Observed_Behaviour_or_Change': str(r['Pattern_Description']),
                'Interpretation': str(r['Interpretation'])
            })

    # Contributing parameters for Faults 4-7
    if contrib_df is not None and not contrib_df.empty:
        for _, r in contrib_df.iterrows():
            pct = float(r['Percent_Change_vs_Normal'])
            direction = 'higher' if pct > 0 else 'lower' if pct < 0 else 'similar'
            rows.append({
                'Section': 'Contributing Parameter',
                'Fault': f"Fault {int(r['Fault'])}",
                'Pattern_or_Rank': f"Rank {int(r['Rank'])}",
                'Samples': '',
                'Parameter': str(r['Sensor']),
                'Observed_Behaviour_or_Change': f"{direction} ({pct:+.3f}% vs normal median)",
                'Interpretation': 'Descriptive data-derived association; not a proven physical fault subtype or causal root cause.'
            })

    # Model summary row
    rows.append({
        'Section': 'Model Summary', 'Fault': 'All classes',
        'Pattern_or_Rank': '', 'Samples': '', 'Parameter': f'Sequence length: {selected_seq} seconds',
        'Observed_Behaviour_or_Change': f'Accuracy {acc:.6f}; Macro Precision {mp:.6f}; Macro Recall {mr:.6f}; Macro F1 {mf:.6f}',
        'Interpretation': 'LSTM learns relationships between sensor sequences and supplied dataset labels.'
    })

    csv_path = os.path.join(OUT, 'FAULT_CLASS_REFERENCE.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print('Clean fault reference created:', csv_path)


def main():
    reset_generated_dirs()

    if not os.path.exists(DATA):
        raise FileNotFoundError(f'Dataset not found: {DATA}')
    df = pd.read_csv(DATA)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    missing = [c for c in SENSORS + ['Fault_Label'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    df = df.sort_values('Timestamp').reset_index(drop=True)

    y = df['Fault_Label'].astype(int).to_numpy()
    ts = df['Timestamp'].astype(str).to_numpy()
    X = df[SENSORS].astype(float).to_numpy()

    n = len(df)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)
    scaler = StandardScaler().fit(X[:tr_end])
    Xs = scaler.transform(X)

    # ------------------------------------------------------------
    # 1. Controlled sequence-length experiment: 10, 20, 30 sec
    # ------------------------------------------------------------
    sequence_rows = []
    candidates = {}
    for seq_len in SEQ_LENGTHS:
        print('\n' + '=' * 72)
        print(f'SEQUENCE LENGTH = {seq_len} seconds')
        print('=' * 72)
        model, history, val_f1, test_acc, test_f1, Xte, yte, tte, test_prob = train_one(
            seq_len, Xs, y, ts, tr_end, va_end
        )
        sequence_rows.append({
            'Sequence_Length': seq_len,
            'Validation_Macro_F1': val_f1,
            'Test_Accuracy': test_acc,
            'Test_Macro_F1': test_f1,
            'Epochs_Trained': len(history.history.get('loss', []))
        })
        candidates[seq_len] = (model, history, Xte, yte, tte, test_prob)
        print(f'Validation Macro F1: {val_f1:.6f}')
        print(f'Test Accuracy: {test_acc:.6f}')
        print(f'Test Macro F1: {test_f1:.6f}')

    seq_table = pd.DataFrame(sequence_rows)
    seq_table.to_csv(os.path.join(ANALYSIS, 'sequence_length_comparison.csv'), index=False)

    selected_seq = int(seq_table.loc[seq_table['Validation_Macro_F1'].idxmax(), 'Sequence_Length'])
    model, history, Xte, yte, tte, test_prob = candidates[selected_seq]
    pred = test_prob.argmax(axis=1)
    conf = test_prob.max(axis=1)

    print('\n' + '=' * 72)
    print(f'SELECTED SEQUENCE LENGTH: {selected_seq} seconds')
    print('=' * 72)

    # ------------------------------------------------------------
    # 2. Save model + training information
    # ------------------------------------------------------------
    model.save(os.path.join(MODEL_DIR, 'final_lstm_model.keras'))
    pd.DataFrame(history.history).to_csv(os.path.join(ANALYSIS, 'training_history.csv'), index=False)

    # ------------------------------------------------------------
    # 3. Final classification results
    # ------------------------------------------------------------
    acc = accuracy_score(yte, pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        yte, pred, labels=list(range(8)), average='macro', zero_division=0
    )

    report = pd.DataFrame(classification_report(
        yte, pred, labels=list(range(8)), target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )).T
    report.to_csv(os.path.join(OUT, 'classification_report.csv'))

    cm = pd.DataFrame(
        confusion_matrix(yte, pred, labels=list(range(8))),
        index=[f'Actual_{i}' for i in range(8)],
        columns=[f'Pred_{i}' for i in range(8)]
    )
    cm.to_csv(os.path.join(OUT, 'confusion_matrix.csv'))

    pred_df = pd.DataFrame({
        'Timestamp': tte,
        'Actual_Label': yte,
        'Predicted_Label': pred,
        'Confidence': conf
    })
    pred_df.to_csv(os.path.join(ANALYSIS, 'test_predictions.csv'), index=False)

    # ------------------------------------------------------------
    # 4. Extract learned 32-D LSTM representation
    # ------------------------------------------------------------
    # Keras 3 Sequential models may not expose .input until explicitly called.
    # The model is already built through Input(), but we explicitly call it once
    # to make the graph/node available across Keras/TensorFlow versions.
    _ = model(Xte[:1], training=False)
    embed = Model(inputs=model.inputs, outputs=model.get_layer('embedding').output)
    emb = embed.predict(Xte, verbose=0)
    emb_df = pd.DataFrame(emb, columns=[f'LSTM_Feature_{i+1}' for i in range(32)])
    emb_df['Timestamp'] = tte
    emb_df['Actual_Label'] = yte
    emb_df['Predicted_Label'] = pred

    # ------------------------------------------------------------
    # 5. Validated pattern discovery: only Faults 1-3
    # ------------------------------------------------------------
    raw_test = df.iloc[va_end:].copy().reset_index(drop=True)
    raw_test['Timestamp'] = raw_test['Timestamp'].astype(str)
    merged = emb_df.merge(raw_test, on=['Timestamp'], how='inner', suffixes=('', '_raw'))
    # Keep the LSTM/evaluation label as the canonical label for downstream analysis.
    merged['Actual_Label'] = merged['Actual_Label'].astype(int)

    pattern_rows = []
    assign_rows = []
    sensor_rows = []
    validated_k = {1: 2, 2: 3, 3: 2}
    normal = merged[merged['Actual_Label'] == 0]

    for fault, k in validated_k.items():
        sub = merged[merged['Actual_Label'] == fault].copy()
        if len(sub) < k:
            continue
        Z = sub[[f'LSTM_Feature_{i+1}' for i in range(32)]].to_numpy()
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(Z)
        sub['Pattern'] = km.labels_ + 1

        for _, row in sub[['Timestamp', 'Actual_Label', 'Predicted_Label', 'Pattern']].iterrows():
            assign_rows.append(row.to_dict())

        for cluster_id in range(1, k + 1):
            cluster = sub[sub['Pattern'] == cluster_id]
            top = sensor_profile(cluster, normal, top_n=5)
            desc = '; '.join(
                f'{s} ({"higher" if pct > 0 else "lower" if pct < 0 else "similar"}, {pct:+.1f}%)'
                for s, _, _, pct in top
            )
            pattern_rows.append({
                'Fault': fault,
                'Pattern': cluster_id,
                'Samples': len(cluster),
                'Top_Parameters': ', '.join(x[0] for x in top),
                'Pattern_Description': desc,
                'Interpretation': 'Descriptive LSTM-representation behavioural pattern; not a proven physical fault subtype or causal root cause.'
            })
            for sensor in SENSORS:
                nm = float(normal[sensor].median())
                cm = float(cluster[sensor].median())
                pct = ((cm - nm) / nm * 100) if nm != 0 else np.nan
                sensor_rows.append({
                    'Fault': fault,
                    'Pattern': cluster_id,
                    'Sensor': sensor,
                    'Normal_Median': nm,
                    'Pattern_Median': cm,
                    'Percent_Change_vs_Normal': pct
                })

    patterns = pd.DataFrame(pattern_rows)
    pd.DataFrame(assign_rows).to_csv(os.path.join(ANALYSIS, 'pattern_cluster_assignments.csv'), index=False)
    pd.DataFrame(sensor_rows).to_csv(os.path.join(ANALYSIS, 'pattern_sensor_statistics.csv'), index=False)

    # ------------------------------------------------------------
    # 6. Contributing-parameter profiles for Faults 4-7
    # ------------------------------------------------------------
    contrib = []
    for fault in range(4, 8):
        sub = merged[merged['Actual_Label'] == fault]
        top = sensor_profile(sub, normal, top_n=5)
        for rank, (sensor, nm, fm, pct) in enumerate(top, 1):
            contrib.append({
                'Fault': fault,
                'Rank': rank,
                'Sensor': sensor,
                'Normal_Median': nm,
                'Fault_Median': fm,
                'Percent_Change_vs_Normal': pct
            })
    contrib_df = pd.DataFrame(contrib)
    contrib_df.to_csv(os.path.join(OUT, 'contributing_parameters.csv'), index=False)

    # ------------------------------------------------------------
    # 7. Clean human-readable fault reference
    # ------------------------------------------------------------
    generate_fault_class_reference(patterns, contrib_df, selected_seq, acc, mp, mr, mf)

    print('\n' + '=' * 72)
    print('FINAL CLEAN PIPELINE COMPLETED')
    print('=' * 72)
    print(f'Selected sequence: {selected_seq} seconds')
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro Precision: {mp:.4f}')
    print(f'Macro Recall: {mr:.4f}')
    print(f'Macro F1: {mf:.4f}')
    print('Main outputs:', sorted(os.listdir(OUT)))
    print('Intermediate outputs:', sorted(os.listdir(ANALYSIS)))
    print('Model:', os.path.join(MODEL_DIR, 'final_lstm_model.keras'))


if __name__ == '__main__':
    main()
