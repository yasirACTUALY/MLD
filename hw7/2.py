import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Function to minimize: x^2 + 2y^2 + 2sin(2πx)sin(2πy)"""
    return x**2 + 2*y**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def gradient(x, y):
    """Gradient of the function"""
    dx = 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    dy = 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    return np.array([dx, dy])

def gradient_descent(x,y,learning_rate, iterations=50):
    
    # Store function values for plotting
    function_values = []
    minVal = f(x,y)
    
    # Perform gradient descent
    for _ in range(iterations):
        # Store current function value
        function_values.append(f(x, y))
        
        # Calculate gradient
        grad = gradient(x, y)
        
        # Update x and y
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        if f(x,y) < minVal:
            minVal = f(x,y)
    
    # Add final function value
    function_values.append(f(x, y))
    
    return np.array(function_values), (x, y), minVal

# Run gradient descent with different learning rates
iterations = 50
lr_small = 0.01
lr_large = 0.1
x = 0.1
y=0.1

values_small, final_point_small, min_val_small = gradient_descent(x,y,lr_small)
values_large, final_point_large, min_val_large = gradient_descent(x,y,lr_large)
print(f"{lr_small} & ({x}, {y}) & {min_val_small:.6f} & ({final_point_small[0]:.6f}, {final_point_small[1]:.6f})\\\\")
print(f"{lr_large} & ({x}, {y}) & {min_val_large:.6f} & ({final_point_large[0]:.6f}, {final_point_large[1]:.6f})\\\\")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(range(len(values_small)), values_small, 'b-', label=f'η = {lr_small}')
plt.plot(range(len(values_large)), values_large, 'r-', label=f'η = {lr_large}')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.grid(True)
plt.show()

# Print final values
print(f"For η = {lr_small}:")
print(f"Final point: x = {final_point_small[0]:.6f}, y = {final_point_small[1]:.6f}")
print(f"Final value: {values_small[-1]:.6f}\n")

print(f"For η = {lr_large}:")
print(f"Final point: x = {final_point_large[0]:.6f}, y = {final_point_large[1]:.6f}")
print(f"Final value: {values_large[-1]:.6f}")

# Visualize the function landscape
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='Function Value')
plt.plot(final_point_small[0], final_point_small[1], 'b*', label=f'Final Point (η = {lr_small})')
plt.plot(final_point_large[0], final_point_large[1], 'r*', label=f'Final Point (η = {lr_large})')
plt.plot(0.1, 0.1, 'g*', label='Starting Point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Contour Plot with Gradient Descent Results')
plt.legend()
plt.grid(True)
plt.show()