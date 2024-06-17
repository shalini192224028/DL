import numpy as np
import matplotlib.pyplot as plt
  
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
  
# Define the range of values for z
z_values = np.arange(-5, 5, 0.1)

# Compute the sigmoid of these values
sigmoid_values = sigmoid(z_values)

# Plot the sigmoid function
plt.plot(z_values, sigmoid_values)
plt.title('Visualization of the Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()
