# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Step 1: Load the dataset
file_path = 'cleaned_combined_data.csv'  # Ensure the correct file path
data = pd.read_csv(file_path)

# Step 2: Filter rows where Element == "Yield"
filtered_data = data[data['Element'] == 'Yield'].drop(columns=['Element'])

# Step 3: Encode the 'Item' column (Crop Type)
label_encoder = LabelEncoder()
filtered_data['Item'] = label_encoder.fit_transform(filtered_data['Item'])

# Step 4: Handle Outliers using IQR
numerical_columns = ['Value', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR_SUM',
                     'RH2M', 'WS2M', 'GWETTOP', 'GWETROOT', 'GWETPROF']

Q1 = filtered_data[numerical_columns].quantile(0.25)
Q3 = filtered_data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
filtered_data = filtered_data[~((filtered_data[numerical_columns] < (Q1 - 1.5 * IQR)) |
                                (filtered_data[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 5: Normalize data using StandardScaler (better for stability)
scaler = StandardScaler()
filtered_data[numerical_columns] = scaler.fit_transform(filtered_data[numerical_columns])

# Step 6: Split the dataset into training and testing sets (chronological order)
train_size = int(len(filtered_data) * 0.8)  # 80% for training
train_data = filtered_data.iloc[:train_size]
test_data = filtered_data.iloc[train_size:]

# Save the processed data (optional)
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Step 7: Prepare the data for LSTM
target_column = 'Value'
X_train = train_data.drop(columns=[target_column]).values
y_train = train_data[target_column].values
X_test = test_data.drop(columns=[target_column]).values
y_test = test_data[target_column].values

# Reshape inputs to 3D for LSTM: [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Step 8: Improved LSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True, activation='relu'),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Step 9: Set up Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Step 10: Train the model
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=16,  # Increased batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler],  # Include callbacks
    verbose=1
)

# Step 11: Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# Step 12: Make predictions
y_pred = model.predict(X_test)

# Step 13: Visualize results (optional)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', marker='o')
plt.plot(y_pred, label='Predicted Values', marker='x')
plt.title('Actual vs Predicted Crop Yield')
plt.xlabel('Sample Index')
plt.ylabel('Yield (Normalized)')
plt.legend()
plt.grid()
plt.show()

# Step 14: Save the model in TensorFlow's recommended format
model.save('crop_yield_model.keras')

print("✅ Model saved successfully!")
