# Import necessary libraries
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import BayesianOptimization, HyperModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import MeanSquaredError

# Load the sequence data from CSV
data = pd.read_csv('ATLNJ1981-2022.csv', header=0, parse_dates=[0])

# Extract 'Date' and 'Time (GMT)' columns from the dataframe
DateTime = data[['Date', 'Time (GMT)']]
data = data.drop(columns=['Date', 'Time (GMT)'])

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

# Define the number of input and output features
num_input_features = X_norm.shape[1]
num_output_features = 1

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:train_size], X_norm[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

lookback = 24

# Function to create RNN dataset
def create_rnn_dataset(X, y, lookback):
    X_rnn, y_rnn = [], []
    for i in range(len(X) - lookback):
        X_rnn.append(X[i:(i + lookback)])
        y_rnn.append(y[i + lookback])
    return np.array(X_rnn), np.array(y_rnn)

# Create RNN datasets for training and testing
X_train_rnn, y_train_rnn = create_rnn_dataset(X_train, y_train, lookback)
X_test_rnn, y_test_rnn = create_rnn_dataset(X_test, y_test, lookback)


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
        
        model.add(CustomAttentionLayer(emphasis_factor=hp.Float('emphasis_factor', min_value=1.0, max_value=2.0, step=0.1)))    # Add attention layer here

        # Attention outputs (batch, 1, features)—flatten before Dense
        model.add(tf.keras.layers.Flatten())
        
        model.add(Dense(num_output_features))
        model.compile(loss=MeanAbsoluteError(), optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")), metrics=[MeanSquaredError()])
        return model

# Bayesian Optimization setup
tuner = BayesianOptimization(
    hypermodel = LSTMHyperModel(lookback=lookback, n_features=num_input_features),
    objective="val_loss",
    max_trials=50,  # Adjust the number of trials as needed
    seed=42,
    project_name="Trials"
)

# EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Perform Bayesian Optimization on the training data
tuner.search(X_train_rnn, y_train_rnn, epochs=500, batch_size=128, callbacks=[early_stopping], validation_split=0.3)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first, second, and third LSTM layers are {best_hps.get('units_1')}, {best_hps.get('units_2')}, and {best_hps.get('units_3')}, respectively.
The optimal dropout rates are {best_hps.get('dropout_1')}, {best_hps.get('dropout_2')}, and {best_hps.get('dropout_3')}. 
The optimal learning rate is {best_hps.get('learning_rate')}.
The optimal emphasis factor for the attention layer is {best_hps.get('emphasis_factor')}.
""")

import matplotlib.pyplot as plt

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
plt.savefig("AC128b24hMAE.png", bbox_inches='tight', dpi=300)
plt.show()

# Time Series Cross-Validation for Model Evaluation
tscv = TimeSeriesSplit(n_splits=10)
model_evaluation_scores = []

for train_index, val_index in tscv.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    X_train_fold_rnn, y_train_fold_rnn = create_rnn_dataset(X_train_fold, y_train_fold, lookback)
    X_val_fold_rnn, y_val_fold_rnn = create_rnn_dataset(X_val_fold, y_val_fold, lookback)

    # Build and train the model using the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train_fold_rnn, y_train_fold_rnn, epochs=500, batch_size=128, callbacks=[early_stopping], validation_data=(X_val_fold_rnn, y_val_fold_rnn))
    
    # Evaluate the model on the validation set of the current fold
    scores = model.evaluate(X_val_fold_rnn, y_val_fold_rnn, verbose=0)
    model_evaluation_scores.append(scores)

# Calculate and print the average performance across all folds
average_performance = np.mean(model_evaluation_scores, axis=0)
print(f"Average Validation Performance: {average_performance}")

# Train the final model using the entire training data and best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train_rnn, y_train_rnn, epochs=500, batch_size=128, callbacks=[early_stopping], validation_split=0.2)

# Save the final model
best_model.save("AC128b24hMAECV.h5")
loaded_model = tf.keras.models.load_model("AC128b24hMAECV.h5", custom_objects={'CustomAttentionLayer': CustomAttentionLayer})

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
plt.savefig("AC128b24hMAEtrainnvalidloss.png", bbox_inches='tight', dpi=300)
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
plt.savefig("AC128b24hMAE.png", bbox_inches='tight', dpi=300)
plt.show()

# Assuming the 'DateTime' DataFrame has the same length as 'y_original' and 'test_predictions_original'
DateTime_indexed = DateTime.iloc[train_size+lookback:]
        
data_full = pd.DataFrame({
    'Date': DateTime_indexed['Date'],
    'Time (GMT)': DateTime_indexed['Time (GMT)'],
    'Original Data':  y_original[train_size+lookback:].flatten(),
    'Test Predictions': test_predictions_original.flatten()
})
        
# Display the DataFrame
print(data_full)
        
# Save the DataFrame as a CSV file
data_full.to_csv('AC128b24hMAE.csv', index=False)     
print("AC128b24hMAE.csv")

# Extract the original data and test predictions from data_full
orig_data = data_full['Original Data'].to_numpy()
test_predictions_full = data_full['Test Predictions'].to_numpy()

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(orig_data, test_predictions_full)
print("Mean Squared Error (MSE):", mse)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(orig_data, test_predictions_full)
print("Mean Absolute Error (MAE):", mae)

# Calculate the R-squared (Coefficient of Determination)
r2 = r2_score(orig_data, test_predictions_full)
print("R-squared (Coefficient of Determination):", r2)

# Calculate the correlation coefficient
correlation_coeff, _ = pearsonr(orig_data, test_predictions_full)
print("Correlation Coefficient:", correlation_coeff)

# Calculate the Mean Absolute Scaled Error (MASE)
naive_forecast = orig_data[:-1]
actual_values = orig_data[1:]
naive_mae = mean_absolute_error(actual_values, naive_forecast)
mase = mae / naive_mae
print("Mean Absolute Scaled Error (MASE):", mase)

# Visual inspection
plt.figure(figsize=(6, 5))
plt.plot(orig_data, marker='o', linestyle='', label="Original Data")
plt.plot(test_predictions_full, marker='o', linestyle='', label="Test Predictions")
plt.xlabel("Time")
plt.ylabel("Water Level")
plt.legend()
plt.savefig("AC128b24hMAE.png", bbox_inches='tight', dpi=300)
plt.show()

# Extract the original data and test predictions from data_full
orig_data = data_full['Original Data'].to_numpy()
test_predictions_full = data_full['Test Predictions'].to_numpy()

# Calculate the linear regression parameters and R-squared value
slope, intercept, r_value, p_value, std_err = linregress(orig_data.flatten(), test_predictions_full.flatten())
r_squared = r_value ** 2

plt.figure(figsize=(6, 6))
# plt.scatter(orig_data, test_predictions_full, alpha=0.5, label="Data Points")

# Plot Original Data and Test Predictions as separate points with different colors
plt.scatter(orig_data, orig_data, color='blue', label="Original Data")
plt.scatter(orig_data, test_predictions_full, color='orange', label=f"Test Predictions (R-squared: {r_squared:.2f})")
plt.xlabel("Original Data")
plt.ylabel("Test Predictions")

# Add a diagonal line (one-to-one line) that represents perfect predictions
min_value = min(min(orig_data), min(test_predictions_full))
max_value = max(max(orig_data), max(test_predictions_full))
plt.plot([min_value, max_value], [min_value, max_value], 'r--')

plt.legend()
plt.savefig("AC128b24hMAEorigvspred.png", bbox_inches='tight', dpi=300)
plt.show()
