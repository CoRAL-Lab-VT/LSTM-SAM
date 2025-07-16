# Import the necessary libraries
import os
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Environment Setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)
os.environ["OMP_NUM_THREADS"] = "20"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print('GPU is being used')
    except RuntimeError as e:
        print(e)

# Load Data
data = pd.read_csv('sandy_1992.csv', header=0, parse_dates=[0])
DateTime = data[['Date', 'Time (GMT)']]
data = data.drop(columns=['Date', 'Time (GMT)'])
print("Target station data imported! Loading LSTM-SAM models...")

# Output directory
location = "Visualization"
os.makedirs(location, exist_ok=True)

# Preprocess Data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def normalize_data(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

X_norm, scaler = normalize_data(X)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_reshaped = y.values.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y_reshaped)

def create_rnn_dataset(X_norm, y_scaled, lookback):
    Xs, ys = [], []
    for i in range(len(X_norm) - lookback):
        Xs.append(X_norm[i:(i + lookback)])
        ys.append(y_scaled[i + lookback])
    return np.array(Xs), np.array(ys)

import tensorflow as tf
from tensorflow.keras import backend as K

class CustomAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, emphasis_factor=1.5, **kwargs):
        self.emphasis_factor = emphasis_factor
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"emphasis_factor": self.emphasis_factor})
        return config

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, T, features)
        e = K.tanh(K.dot(x, self.W) + self.b)        
        a = K.softmax(e, axis=1)                    

        # Squeeze out the features‐singleton so we have (batch, T)
        a_flat = tf.squeeze(a, axis=-1)              

        # Compute the top 10% of T
        T = tf.shape(a_flat)[1]
        k = tf.maximum(1, tf.cast(tf.cast(T, tf.float32) * 0.1, tf.int32))

        # Now pick top-k over *time* (axis=-1 here means the last axis of a_flat)
        top_vals, top_idx = tf.nn.top_k(a_flat, k)   # both (batch, k)

        # Build a (batch, T) mask where True for any time-step in top_idx
        top_idx_exp = tf.expand_dims(top_idx, axis=2)       
        range_row   = tf.reshape(tf.range(T), (1, 1, T))     
        mask_flat   = tf.reduce_any(
            tf.equal(top_idx_exp, range_row), axis=1          
        )

        # Re-expand to (batch, T, 1) to match a’s shape
        mask = tf.expand_dims(mask_flat, axis=-1)            

        # Emphasize only those top-10% time-steps
        a_emph = tf.where(mask, a * self.emphasis_factor, a)  

        # Final weighted sum over time
        output = tf.reduce_sum(x * a_emph, axis=1, keepdims=True)  
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[-1])

def kge(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def nse(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - np.mean(y_true))**2)
    return 1 - num/den

def amb(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def willmott_skill(y_true, y_pred):
    obs_mean = np.mean(y_true)
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((np.abs(y_pred - obs_mean) + np.abs(y_true - obs_mean))**2)
    return 1 - num/den

def evaluate_and_plot(model, X_test_rnn, y_test_rnn, dates, scaler_y, filename_prefix):
    # Evaluate and predict
    _ = model.evaluate(X_test_rnn, y_test_rnn, verbose=1)
    preds = model.predict(X_test_rnn)

    # Inverse scale
    y_true = scaler_y.inverse_transform(y_test_rnn)
    y_pred = scaler_y.inverse_transform(preds)

    # By default, use all data
    y_test_filtered = y_true
    test_predictions_filtered = y_pred

    # Limit to a specific period, comment out the lines below for the full timeseries prediction:
    start_date = pd.to_datetime('1992-12-08')
    end_date   = pd.to_datetime('1992-12-14')
    mask = (dates >= start_date) & (dates <= end_date)
    y_test_filtered = y_true[mask.values]
    test_predictions_filtered = y_pred[mask.values]

    # Compute metrics on the filtered subset
    rmse = np.sqrt(mean_squared_error(y_test_filtered, test_predictions_filtered))
    kge_v = kge(y_test_filtered.flatten(), test_predictions_filtered.flatten())
    nse_v = nse(y_test_filtered.flatten(), test_predictions_filtered.flatten())
    d_v = willmott_skill(y_test_filtered.flatten(), test_predictions_filtered.flatten())
    amb_v = amb(y_test_filtered.flatten(), test_predictions_filtered.flatten())

    metrics_text = [
        f'RMSE: {rmse:.4f}', f'KGE: {kge_v:.4f}',
        f'NSE: {nse_v:.4f}', f"Willmott's d: {d_v:.4f}",
        f'mBias: {amb_v:.4f}'
    ]

    # Plot both subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time series plot (no legend)
    ax1.plot(y_test_filtered, 'r--', label='Actual')
    ax1.plot(test_predictions_filtered, label='Predicted')
    for text in metrics_text:
        ax1.plot([], [], ' ', label=text)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Water level (m)')
    ax1.set_title(f'Sandy hook, NJ ({filename_prefix})')

    # One-to-one scatter with legend
    mn, mx = min(y_test_filtered.min(), test_predictions_filtered.min()), \
             max(y_test_filtered.max(), test_predictions_filtered.max())
    ax2.plot([mn, mx], [mn, mx], 'k--', label='1:1 line')
    ax2.scatter(y_test_filtered, test_predictions_filtered, alpha=0.7, label='Predicted')
    ax2.scatter(y_test_filtered, y_test_filtered, s=5, label='Actual')
    for text in metrics_text:
        ax2.plot([], [], ' ', label=text)
    ax2.set_xlabel('Actual (m)')
    ax2.set_ylabel('Predicted (m)')
    ax2.set_title(f'Sandy hook, NJ ({filename_prefix})')
    ax2.legend(loc='best')

    plt.tight_layout()
    fig.savefig(f"{location}/{filename_prefix}_combined.png", dpi=300, bbox_inches='tight')
    plt.show()

    return rmse, kge_v, nse_v, d_v, amb_v

def extract_lookback_from_filename(filename):
    h_pos = filename.find('h')
    if h_pos > 0 and filename[h_pos-1].isdigit():
        start = h_pos-2 if h_pos>1 and filename[h_pos-2].isdigit() else h_pos-1
        return int(filename[start:h_pos])
    return None

def extract_prefix_from_filename(filename):
    return filename.split(".h5")[0]

def extract_and_save_all(test_pred, y_true, lookback, DateTime, prefix, filename):
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    indices = np.arange(len(y_true.flatten()))
    pred_col = prefix.replace("lstm", "")
    df = pd.DataFrame({
        'Actual water level': y_true.flatten()[indices],
        pred_col: test_pred.flatten()[indices]
    })

    idx_adj = indices + lookback
    df['Date'] = DateTime['Date'].iloc[idx_adj].values
    df['Time (GMT)'] = DateTime['Time (GMT)'].iloc[idx_adj].values
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Time (GMT)', 'Actual water level', pred_col]]

    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        existing = pd.read_csv(filename)
        existing['Date'] = pd.to_datetime(existing['Date'])
        existing = existing.drop(columns=[pred_col], errors='ignore')
        merged = existing.merge(
            df[['Date', 'Time (GMT)', pred_col]],
            on=['Date', 'Time (GMT)'], how='left'
        )
        merged.to_csv(filename, index=False)

    return df

# Main loop: load each model, evaluate, plot, and save results
all_files   = os.listdir()
model_files = [f for f in all_files if f.endswith(".h5")]
lookbacks   = [extract_lookback_from_filename(f) for f in model_files]
prefixes    = [extract_prefix_from_filename(f) for f in model_files]

print("Starting predictions...")

# Prepare additional output directory
os.makedirs('LSTM-SAM predictions', exist_ok=True)
filename = "LSTM-SAM predictions/DAT_EWL_SANDY1992.csv"

for model_file, lookback, prefix in zip(model_files, lookbacks, prefixes):
    X_test, y_test = create_rnn_dataset(X_norm, y_scaled, lookback)

    # build dates series for filtering
    dates_test = DateTime['Date'].iloc[lookback:].reset_index(drop=True)

    model = load_model(model_file, custom_objects={'CustomAttentionLayer': CustomAttentionLayer})
    rmse, kge_v, nse_v, d_v, amb_v = evaluate_and_plot(
        model, X_test, y_test, dates_test, scaler_y, prefix
    )

    test_pred_orig = scaler_y.inverse_transform(model.predict(X_test))
    y_test_orig   = scaler_y.inverse_transform(y_test)
    extract_and_save_all(test_pred_orig, y_test_orig, lookback, DateTime, prefix, filename)

print("Predictions completed!")
