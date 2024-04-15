# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from Excel file
data = pd.read_excel(r"C:\Users\SAHITHI\Downloads\one\Folds5x2_pp.xlsx")

# Handling missing values if any
data.dropna(inplace=True)

# Encoding categorical variables if any

# Separate features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

activation_functions = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative)
}
# Define optimizer functions
def sgd_with_momentum(weights, biases, d_weights, d_biases, velocities, learning_rate, momentum):
    for i in range(len(weights)):
        velocities[i] = momentum * velocities[i] - learning_rate * d_weights[i]
        weights[i] += velocities[i]
        biases[i] -= learning_rate * d_biases[i]

def adam(weights, biases, d_weights, d_biases, velocities, squared_gradients, beta1, beta2, learning_rate, t):
    epsilon = 1e-8
    for i in range(len(weights)):
        velocities[i] = beta1 * velocities[i] + (1 - beta1) * d_weights[i]
        squared_gradients[i] = beta2 * squared_gradients[i] + (1 - beta2) * d_weights[i]**2
        velocities_corrected = velocities[i] / (1 - beta1**t)
        squared_gradients_corrected = squared_gradients[i] / (1 - beta2**t)
        weights[i] -= learning_rate * velocities_corrected / (np.sqrt(squared_gradients_corrected) + epsilon)
        biases[i] -= learning_rate * d_biases[i]
# Define MAPE error calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Initialize weights and biases
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    biases_input_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    biases_hidden_output = np.zeros((1, output_size))
    return weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output
# Define forward propagation function
def forward_propagation(X, weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, activation):
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_input_hidden
    if activation == 'sigmoid':
        hidden_layer_output = sigmoid(hidden_layer_input)
    elif activation == 'relu':
        hidden_layer_output = relu(hidden_layer_input)
    elif activation == 'tanh':
        hidden_layer_output = tanh(hidden_layer_input)
    else:
        raise ValueError("Invalid activation function!")
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_hidden_output
    output_layer_output = output_layer_input  # Linear activation for regression
    return output_layer_output, hidden_layer_output

# Define backward propagation function
def backward_propagation(X, y, output_layer_output, hidden_layer_output, weights_hidden_output, activation):
    output_loss = output_layer_output - y
    if activation == 'sigmoid':
        hidden_loss = np.dot(output_loss, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
    elif activation == 'relu':
        hidden_loss = np.dot(output_loss, weights_hidden_output.T) * relu_derivative(hidden_layer_output)
    elif activation == 'tanh':
        hidden_loss = np.dot(output_loss, weights_hidden_output.T) * tanh_derivative(hidden_layer_output)
    else:
        raise ValueError("Invalid activation function!")
    return output_loss, hidden_loss

# Define regularization function
def apply_regularization(weights, lambda_reg):
    return lambda_reg * np.sum(weights**2)# Define training function with early stopping
def train_with_early_stopping(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, activation, optimizer, learning_rate, momentum=None, beta1=None, beta2=None, epochs=1000, batch_size=64, lambda_reg=0, patience=10, min_delta=0.0001):
    weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    velocities = [np.zeros_like(weights_input_hidden), np.zeros_like(weights_hidden_output)]
    squared_gradients = [np.zeros_like(weights_input_hidden), np.zeros_like(weights_hidden_output)]
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    patience_count = 0

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size].reshape(-1, 1)

            # Forward propagation
            output_layer_output, hidden_layer_output = forward_propagation(x_batch, weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, activation)

            # Calculate loss
            loss = np.mean((output_layer_output - y_batch) ** 2)
            reg_loss = loss + apply_regularization(weights_hidden_output, lambda_reg) + apply_regularization(weights_input_hidden, lambda_reg)

            # Backward propagation
            output_loss, hidden_loss = backward_propagation(x_batch, y_batch, output_layer_output, hidden_layer_output, weights_hidden_output, activation)

            # Update weights and biases using gradient descent
            d_weights_hidden_output = np.dot(hidden_layer_output.T, output_loss)
            d_biases_hidden_output = np.sum(output_loss, axis=0, keepdims=True)
            d_weights_input_hidden = np.dot(x_batch.T, hidden_loss)
            d_biases_input_hidden = np.sum(hidden_loss, axis=0, keepdims=True)

            if optimizer == 'sgd_with_momentum':
                sgd_with_momentum([weights_input_hidden, weights_hidden_output], [biases_input_hidden, biases_hidden_output], [d_weights_input_hidden, d_weights_hidden_output], [d_biases_input_hidden, d_biases_hidden_output], velocities, learning_rate, momentum)
            elif optimizer == 'adam':
                t = epoch * (len(X_train) // batch_size) + i // batch_size + 1
                adam([weights_input_hidden, weights_hidden_output], [biases_input_hidden, biases_hidden_output], [d_weights_input_hidden, d_weights_hidden_output], [d_biases_input_hidden, d_biases_hidden_output], velocities, squared_gradients, beta1, beta2, learning_rate, t)
            else:
                raise ValueError("Invalid optimizer!")

        # Calculate training loss
        output_layer_output, _ = forward_propagation(X_train, weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, activation)
        train_loss = np.mean((output_layer_output - y_train.reshape(-1, 1)) ** 2)
        reg_train_loss = train_loss + apply_regularization(weights_hidden_output, lambda_reg) + apply_regularization(weights_input_hidden, lambda_reg)
        train_losses.append(reg_train_loss)

        # Calculate validation loss
        output_layer_output, _ = forward_propagation(X_val, weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, activation)
        val_loss = np.mean((output_layer_output - y_val.reshape(-1, 1)) ** 2)
        reg_val_loss = val_loss + apply_regularization(weights_hidden_output, lambda_reg) + apply_regularization(weights_input_hidden, lambda_reg)
        val_losses.append(reg_val_loss)
        
        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

        # Early stopping
        if epoch > 0 and val_losses[-1] - min_delta < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch + 1} with validation loss: {val_losses[-1]}')
                break

    return weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, train_losses, val_losses
# Train the model with early stopping
input_size = X_train_scaled.shape[1]
hidden_size = 128
output_size = 1
activation = 'sigmoid'
optimizer = 'adam'
learning_rate = 0.001
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epochs = 1000
batch_size = 64
lambda_reg = 0.001
patience = 10
min_delta = 0.0001
activation_functions = [('relu', 'sigmoid')]  # Example: relu for hidden layer and sigmoid for output layer

weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, train_losses, val_losses = train_with_early_stopping(X_train_scaled, y_train, X_val_scaled, y_val, input_size, hidden_size, output_size, activation, optimizer, learning_rate, momentum=momentum, beta1=beta1, beta2=beta2, epochs=epochs, batch_size=batch_size, lambda_reg=lambda_reg, patience=patience, min_delta=min_delta)
# Evaluate the model on the test set
output_layer_output, _ = forward_propagation(X_test_scaled, weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output, activation)
test_loss = np.mean((output_layer_output - y_test.reshape(-1, 1)) ** 2)
print(f'Test Loss: {test_loss}')

# Calculate MAPE error
mape_error = mean_absolute_percentage_error(y_test, output_layer_output.flatten())
print(f'MAPE Error: {mape_error}')

# Save the computed weights
np.savez("model_weights.npz", weights_input_hidden=weights_input_hidden, biases_input_hidden=biases_input_hidden, weights_hidden_output=weights_hidden_output, biases_hidden_output=biases_hidden_output)

           
# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual data vs. predicted data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, output_layer_output.flatten(), color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Ideal Line')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Data')
plt.legend()
plt.grid(True)
plt.show()