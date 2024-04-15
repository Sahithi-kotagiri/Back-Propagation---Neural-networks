import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
x_train = np.linspace(-2*np.pi, 2*np.pi, 1000)
y_train = np.sin(x_train)

x_val = np.random.uniform(-2*np.pi, 2*np.pi, 300)
y_val = np.sin(x_val)

# Normalize inputs and outputs
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 2 - 1

x_train_normalized = normalize_data(x_train)
y_train_normalized = normalize_data(y_train)
x_val_normalized = normalize_data(x_val)
# Define activation functions
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Define neural network parameters
input_layer_size = 1
hidden_layer1_size = 32
hidden_layer2_size = 16
output_layer_size = 1
learning_rate = 0.001
num_epochs = 1000
batch_size = 64
momentum = 0.9
l2_lambda = 0.1

# Initialize weights
np.random.seed(1)
weights_input_hidden1 = np.random.randn(input_layer_size, hidden_layer1_size)
weights_hidden1_hidden2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
weights_hidden2_output = np.random.randn(hidden_layer2_size, output_layer_size)

# Initialize momentum terms
momentum_input_hidden1 = np.zeros_like(weights_input_hidden1)
momentum_hidden1_hidden2 = np.zeros_like(weights_hidden1_hidden2)
momentum_hidden2_output = np.zeros_like(weights_hidden2_output)

# Lists to store training and validation errors
train_errors = []
val_errors = []

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        # Forward pass
        x_batch = x_train_normalized[i:i+batch_size]
        y_batch = y_train_normalized[i:i+batch_size]
        
        hidden_layer1_input = np.dot(x_batch[:, np.newaxis], weights_input_hidden1)
        hidden_layer1_output = tanh(hidden_layer1_input)
        
        hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
        hidden_layer2_output = tanh(hidden_layer2_input)
        
        output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
        output_layer_output = output_layer_input

        # Backward pass
        output_error = output_layer_output - y_batch[:, np.newaxis]
        output_delta = output_error * tanh_derivative(output_layer_input)
        
        hidden2_error = np.dot(output_delta, weights_hidden2_output.T)
        hidden2_delta = hidden2_error * tanh_derivative(hidden_layer2_input)
        
        hidden1_error = np.dot(hidden2_delta, weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * tanh_derivative(hidden_layer1_input)

        # Update weights
        d_weights_hidden2_output = np.dot(hidden_layer2_output.T, output_delta) / len(x_batch)
        d_weights_hidden1_hidden2 = np.dot(hidden_layer1_output.T, hidden2_delta) / len(x_batch)
        d_weights_input_hidden1 = np.dot(x_batch[:, np.newaxis].T, hidden1_delta) / len(x_batch)
        
        momentum_hidden2_output = momentum * momentum_hidden2_output + learning_rate * (d_weights_hidden2_output + l2_lambda * weights_hidden2_output).T
        momentum_hidden1_hidden2 = momentum * momentum_hidden1_hidden2 + learning_rate * (d_weights_hidden1_hidden2 + l2_lambda * weights_hidden1_hidden2)
        momentum_input_hidden1 = momentum * momentum_input_hidden1 + learning_rate * (d_weights_input_hidden1 + l2_lambda * weights_input_hidden1)
        
        weights_hidden2_output -= learning_rate * (np.dot(hidden_layer2_output.T, output_delta) + l2_lambda * weights_hidden2_output)
        weights_hidden1_hidden2 -= learning_rate * (np.dot(hidden_layer1_output.T, hidden2_delta) + l2_lambda * weights_hidden1_hidden2)
        weights_input_hidden1 -= learning_rate * (np.dot(x_batch[:, np.newaxis].T, hidden1_delta) + l2_lambda * weights_input_hidden1)

        
    # Calculate validation error
    hidden_layer1_output_val = tanh(np.dot(x_val_normalized[:, np.newaxis], weights_input_hidden1))
    hidden_layer2_output_val = tanh(np.dot(hidden_layer1_output_val, weights_hidden1_hidden2))
    output_layer_output_val = np.dot(hidden_layer2_output_val, weights_hidden2_output)
    
    val_error = np.mean((output_layer_output_val - y_val[:, np.newaxis])**2)
    val_errors.append(val_error)
    #print(f"Epoch {epoch+1}, Validation Error: {val_error}")

    # Calculate training error
    hidden_layer1_output_train = tanh(np.dot(x_train_normalized[:, np.newaxis], weights_input_hidden1))
    hidden_layer2_output_train = tanh(np.dot(hidden_layer1_output_train, weights_hidden1_hidden2))
    output_layer_output_train = np.dot(hidden_layer2_output_train, weights_hidden2_output)

    train_error = np.mean((output_layer_output_train - y_train[:, np.newaxis])**2)
    train_errors.append(train_error)

    print(f"Epoch {epoch+1}, Validation Error: {val_error}, Training Error: {train_error}")


# Plot results
x_plot = np.linspace(-2*np.pi, 2*np.pi, 1000)
x_plot_normalized = normalize_data(x_plot)
hidden_layer1_output_plot = tanh(np.dot(x_plot_normalized[:, np.newaxis], weights_input_hidden1))
hidden_layer2_output_plot = tanh(np.dot(hidden_layer1_output_plot, weights_hidden1_hidden2))
output_layer_output_plot = np.dot(hidden_layer2_output_plot, weights_hidden2_output)

plt.figure()
plt.plot(x_train, y_train, label='Training Data')
plt.plot(x_plot, output_layer_output_plot, label='ANN Output')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, num_epochs + 1), val_errors, label='Validation Error')
plt.plot(range(1, num_epochs + 1), train_errors, label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Training and Validation Error')
plt.legend()
plt.show()

