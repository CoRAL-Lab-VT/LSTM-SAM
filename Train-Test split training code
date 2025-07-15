# Import the necessary libraries
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import tensorflow.keras as keras
from keras_tuner import Objective
from keras_tuner import HyperModel
from scipy.stats import linregress
from tensorflow.keras import layers
from keras.layers import Activation
from sklearn.datasets import load_iris
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras_tuner.tuners import BayesianOptimization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the sequence data from CSV
data = pd.read_csv('ATLNJ1981-2022.csv', header=0, parse_dates=[0])

# Extract 'Date' and 'Time (GMT)' columns from the dataframe
DateTime = data[['Date', 'Time (GMT)']]

# Drop the date column
data = data.drop(columns=['Date', 'Time (GMT)'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Fortmyers latest.csv dataset
X = data.iloc[:, :-1]  # All columns except the last column (target variable)
y = data.iloc[:, -1]  # The target variable

def normalize_data(X):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

# Step 1: Normalize the data using MinMaxScaler
X_norm, scaler = normalize_data(X)

# Create a new scaler object for the target variable
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Reshape y to a 2D array since MinMaxScaler expects a 2D input
y_reshaped = y.values.reshape(-1, 1)

# Fit and transform the target variable using the new scaler object
y_scaled = scaler_y.fit_transform(y_reshaped)

# Inverse transform the scaled target variable back to the original scale
y_original = scaler_y.inverse_transform(y_scaled)

print(X_norm.shape)
print(y_scaled.shape)

#Training data has to be sequencial - first 3 weeks
#Training data
train_size = int(0.8 * len(X_norm))

#Number of samples to lookback for each sample
lookback= 24


# Define the number of input features
num_input_features = X_norm.shape[1]

# Define the number of output features
num_output_features = 1

# Function to create RNN dataset
def create_rnn_dataset(X_norm, y_scaled, lookback):
    X, y = [], []
    for i in range(len(X_norm)-lookback):
        X.append(X_norm[i:(i+lookback)])
        y.append(y_scaled[(i+lookback)])
    return np.array(X), np.array(y)

# Split the preprocessed dataset into training and testing sets
X_train = X_norm[:train_size]
X_test = X_norm[train_size:]
y_train = y_scaled[:train_size]
y_test = y_scaled[train_size:]

X_train_rnn, y_train_rnn = create_rnn_dataset(X_train, y_train, lookback)
X_test_rnn, y_test_rnn = create_rnn_dataset(X_test, y_test, lookback)
print(X_train_rnn.shape, X_test_rnn.shape)

from keras.layers import Bidirectional, BatchNormalization
from keras.regularizers import l2

# Custom Attention Layer
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

        # Squeeze out the features‐singleton so we have 
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

class LSTMHyperModel(HyperModel):
    def __init__(self, lookback, n_features):
        # now each sample is (lookback, n_features)
        self.input_shape = (lookback, n_features)

    def build(self, hp):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=hp.Int("units_1", min_value=32, max_value=512, step=32),
                                      activation='tanh',
                                      return_sequences=True,
                                      kernel_regularizer=l2(hp.Float("l2_1", min_value=1e-6, max_value=1e-3, sampling="log")), input_shape=self.input_shape)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int("units_2", min_value=32, max_value=512, step=32),
                                      activation='tanh',
                                      return_sequences=True,
                                      kernel_regularizer=l2(hp.Float("l2_2", min_value=1e-6, max_value=1e-3, sampling="log")))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Bidirectional(LSTM(units=hp.Int("units_3", min_value=32, max_value=512, step=32),
                                      activation='tanh',
                                      return_sequences=True,
                                      kernel_regularizer=l2(hp.Float("l2_3", min_value=1e-6, max_value=1e-3, sampling="log")))))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float("dropout_3", min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(CustomAttentionLayer(emphasis_factor=hp.Float('emphasis_factor', min_value=1.0, max_value=2.0, step=0.1)))  
        
        # Attention outputs (batch, 1, features)—flatten before Dense
        model.add(tf.keras.layers.Flatten())
        
        model.add(Dense(num_output_features))
        model.compile(loss="mae", optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")), metrics=["mae"])
        return model

# Set up the hypermodel
hypermodel = LSTMHyperModel( lookback=lookback, n_features=num_input_features )

# Set up the Keras tuner
tuner = BayesianOptimization(
    hypermodel,
    objective="val_loss",
    max_trials=300,  # Increase the number of trials
    seed=42,
    project_name="Trials"
)

# EarlyStopping object that will monitor the validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                               verbose=1, mode='min')
# Tune the model
tuner.search(X_train_rnn, y_train_rnn, epochs=500, batch_size=256, callbacks=[early_stopping], validation_split=0.3)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first, second, and third LSTM layers are {best_hps.get('units_1')}, {best_hps.get('units_2')}, and {best_hps.get('units_3')}, respectively.
The optimal dropout rates are {best_hps.get('dropout_1')}, {best_hps.get('dropout_2')}, and {best_hps.get('dropout_3')}. 
The optimal learning rate is {best_hps.get('learning_rate')}.
The optimal emphasis factor for the attention layer is {best_hps.get('emphasis_factor')}.
""")

# create a new figure
fig, ax = plt.subplots()
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
# text to display
text = f"""
The hyperparameter search is complete. The optimal number of units in the first, second, and third LSTM layers are {best_hps.get('units_1')}, {best_hps.get('units_2')}, and {best_hps.get('units_3')}, respectively.
The optimal dropout rates are {best_hps.get('dropout_1')}, {best_hps.get('dropout_2')}, and {best_hps.get('dropout_3')}. 
The optimal learning rate is {best_hps.get('learning_rate')}.
The optimal emphasis factor for the attention layer is {best_hps.get('emphasis_factor')}.
"""
ax.text(0.5, 5, text, fontsize=12, ha='center')
plt.savefig("CB256b24hMAE.png", bbox_inches='tight', dpi=300)
plt.show()

# Build the model with the optimal hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train_rnn, y_train_rnn, epochs=500, batch_size=256, verbose=1, callbacks=[early_stopping], validation_split=0.3)

# Save the model
best_model.save("CB256b24hMAETT.h5")

# Evaluate the best model
best_model.evaluate(X_test_rnn, y_test_rnn, verbose=1)

# Load the model
loaded_model = load_model("CB256b24hMAETT.h5", custom_objects={'CustomAttentionLayer': CustomAttentionLayer})

# Make predictions on the testing data
test_predictions = loaded_model.predict(X_test_rnn)

# Visualize the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (m)')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.savefig("CB256b24hMAEtrainnvalidloss.png", bbox_inches='tight', dpi=300)
plt.show()

# Reverse the scaling of the predictions
test_predictions_original = scaler_y.inverse_transform(test_predictions)

# Calculate the evaluation metrics
mse = mean_squared_error(y_original[train_size+lookback:], test_predictions_original)
mae = mean_absolute_error(y_original[train_size+lookback:], test_predictions_original)
rmse = math.sqrt(mean_squared_error(y_original[train_size+lookback:], test_predictions_original))
r2 = r2_score(y_original[train_size+lookback:], test_predictions_original)
print(f'Mean Squared Error: {mse:.4f}')
print(f'RMSE: {rmse:.2f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Visualize the actual water levels and the predictions
plt.figure(figsize=(10, 5))
plt.plot(y_original[train_size+lookback:], label='Actual Water Level')
plt.plot(test_predictions_original, label='Predicted Water Level')
plt.xlabel('Time (h)')
plt.ylabel('Water Level (m)')
plt.title('Actual and Predicted Water Levels')
plt.legend()
plt.savefig("CB256b24hMAE_actual_vs_pred.png", bbox_inches='tight', dpi=300)
plt.show()

# Assemble results into a DataFrame
DateTime_indexed = DateTime.iloc[train_size+lookback:].reset_index(drop=True)

data_results = pd.DataFrame({
    'Date':              DateTime_indexed['Date'],
    'Time (GMT)':        DateTime_indexed['Time (GMT)'],
    'Original Waterlevel':  y_original[train_size+lookback:].flatten(),
    'Predicted Waterlevel': test_predictions_original.flatten()
})

# Display and save the full results
print(data_results.head())
data_results.to_csv('CB256b24hMAETT_full.csv', index=False)
print("CB256b24hMAETT_full.csv")

# Extract arrays for metric calculations
orig = data_results['Original Waterlevel'].to_numpy()
pred = data_results['Predicted Waterlevel'].to_numpy()

# Calculate evaluation metrics
mse  = mean_squared_error(orig, pred)
mae  = mean_absolute_error(orig, pred)
rmse = math.sqrt(mse)
r2   = r2_score(orig, pred)
corr, _ = pearsonr(orig, pred)

# For MASE, use a naive one-step lag forecast on the original series
naive_mae = mean_absolute_error(orig[1:], orig[:-1])
mase      = mae / naive_mae

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Correlation Coefficient: {corr:.4f}")
print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")

# Scatter plot: original vs. predicted with one‐one line
plt.figure(figsize=(6, 6))
plt.scatter(orig, pred, alpha=0.5, label=f"Predictions (R² = {r2:.2f})")
plt.plot([orig.min(), orig.max()], [orig.min(), orig.max()], 'r--', label="Perfect fit")
plt.xlabel("Original Water Level (m)")
plt.ylabel("Predicted Water Level (m)")
plt.title("Original vs. Predicted Water Levels")
plt.legend()
plt.savefig("CB256b24hMAE_orig_vs_pred.png", bbox_inches='tight', dpi=300)
plt.show()

