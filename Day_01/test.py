import numpy as np
import matplotlib.pyplot as plt

# 1. Tiny Synthetic Dataset
# 3 observations, 2 features
X = np.array([[1.0, 2.0], 
              [2.0, 3.0], 
              [3.0, 4.0]])
# Target continuous variable
y = np.array([5.0, 8.0, 11.0]) 

# 2. Manual Dot Product vs NumPy
w = np.array([1.5, 1.0]) # Initial arbitrary weight vector

# Manual calculation for the first observation: (1.0 * 1.5) + (2.0 * 1.0) = 3.5
manual_pred = (X[0][0] * w[0]) + (X[0][1] * w[1])

# NumPy calculation for the first observation
numpy_pred = np.dot(X[0], w)

print(f"Manual prediction for Obs 1: {manual_pred}")
print(f"NumPy prediction for Obs 1:  {numpy_pred}")
assert manual_pred == numpy_pred, "Dot product logic failed."





# 3. Simple Loss Calculation (Mean Squared Error)
def compute_mse_loss(X, y, weights):
    m = len(y)
    predictions = np.dot(X, weights)
    # Compute mean square error
    cost = (1/m) * np.sum(np.square(predictions - y)) 
    return cost

initial_loss = compute_mse_loss(X, y, w)
print(f"Initial Loss: {initial_loss}")

# 4. One Manual Gradient-Descent Update
alpha = 0.09 # Learning rate
m = len(y)

# Step A: Compute predictions
predictions = np.dot(X, w)

# Step B: Compute the error
error = predictions - y

# Step C: Compute the gradient (the steepest slope of the function)
gradient = (1/m) * np.dot(X.T, error) 

# Step D: Update the parameters using the gradient
w_updated = w - (alpha * gradient)

updated_loss = compute_mse_loss(X, y, w_updated)
print(f"Loss after ONE manual update: {updated_loss}")
print(f"Original weights: {w}")
print(f"Updated weights: {w_updated}")



def train_model(X_train, y_train, alpha, iterations):
    # Initialize parameters to zeros
    weights = np.zeros(X_train.shape[1]) 
    loss_history = []
    
    for i in range(iterations):
        print(f"Itration {i} - weight_old: {weights}")
        predictions = np.dot(X_train, weights)
        error = predictions - y_train
        gradient = (1/len(y_train)) * np.dot(X_train.T, error)
        weights = weights - (alpha * gradient)
        print(f"Itration {i} - weight_new: {weights}")
        
        loss_history.append(compute_mse_loss(X_train, y_train, weights))
    
    return weights, loss_history

# Experiment 1: Learning Rate Convergence
w_good_lr, loss_good_lr = train_model(X, y, alpha=0.01, iterations=50)
w_high_lr, loss_high_lr = train_model(X, y, alpha=0.15, iterations=50) # Intentionally high

plt.figure(figsize=(10, 4))
plt.plot(loss_good_lr, label='Good LR (0.01)')
plt.plot(loss_high_lr, label='High LR (0.15) - Overshooting')
plt.title('Experiment 1: Learning Rate Convergence')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True) 
plt.show()

# Experiment 2: Poorly Scaled Features
# Creating features with massive numerical differences
X_poor_scale = np.array([[1.0, 2000.0], 
                         [2.0, 3000.0], 
                         [3.0, 4000.0]])
w_poor, loss_poor = train_model(X_poor_scale, y, alpha=0.01, iterations=50)
print(f"Final Loss with Poor Scaling: {loss_poor[-1]} (Notice the explosion/NaNs)")

# Experiment 3: Baseline Comparison
# Baseline: Predict the mean of y for every observation
mean_y = np.mean(y)
baseline_predictions = np.full(shape=y.shape, fill_value=mean_y)
baseline_loss = (1/len(y)) * np.sum(np.square(baseline_predictions - y))

print(f"Baseline (Mean) Loss: {baseline_loss}")
print(f"Our Learned Model Loss: {loss_good_lr[-1]}")

if loss_good_lr[-1] < baseline_loss:
    print("Decision: Our model successfully beat the constant baseline.")