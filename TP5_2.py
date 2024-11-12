from derivativeTP5_1 import finite_derivative
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rosenbrock_partial_deriv_a(a, b):
    return 2 * (200 * (a ** 3) - 200 * a * b + a - 1)

def rosenbrock_partial_deriv_b(a, b):
    return - 200 * (a ** 2 - b)

def f_rosenbrock(a, b):
    return (1 - a)**2 + 100 *(b - a**2)**2

def numerical_gradient(a, b, h=1e-5):
    f_rosenbrock_b_fixed = lambda x: f_rosenbrock(x, b) # partial funciton with b fixed
    f_rosenbrock_a_fixed = lambda x: f_rosenbrock(a, x) # partial funciton with a fixed

    df1 = finite_derivative(f_rosenbrock_b_fixed, a, h=1e-5)
    df2 = finite_derivative(f_rosenbrock_a_fixed, b, h=1e-5)
    return df1, df2

print(numerical_gradient(3, 5))
print(rosenbrock_partial_deriv_a(3, 5))
print(rosenbrock_partial_deriv_b(3, 5))

def optimize_rosenbrock_numerical(a_init, b_init, learning_rate=0.001, num_iterations=2000):
    args = [a_init, b_init]
    for _ in range(num_iterations):
        current_args_descent = numerical_gradient(args[0], args[1])
        args[0] -= learning_rate * current_args_descent[0]
        args[1] -= learning_rate * current_args_descent[1]
    return args[0], args[1]


# Searching best learning rate and then best initial values and ploting them in png files.
# You can chose print to print in terminal, show to have plt.show for interactive or save for saving in .png file
def search_best_learning_rate_and_initial_values(output_type="save"):
    # Parameters
    learning_rates = [0.1 * 10**-i for i in range(5)]
    distances_learning_rate = []
    a_init, b_init = 3, 5
    
    # Define the distance function
    def distance_to_target(a, b):
        return np.sqrt((a - 1) ** 2 + (b - 1) ** 2)
    
    # Part 1: Variation of Learning Rate
    for lr in learning_rates:
        a_final, b_final = optimize_rosenbrock_numerical(a_init, b_init, learning_rate=lr, num_iterations=2000)
        distance = distance_to_target(a_final, b_final)
        distances_learning_rate.append(distance)
        if output_type == "print":
            print(f"Learning Rate: {lr}, Distance to Target: {distance}")
    
    best_learning_rate = learning_rates[np.argmin(distances_learning_rate)]
    if output_type == "print":
        print(f"\nBest Learning Rate: {best_learning_rate}")
    
    # Plotting the learning rate variation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.log10(learning_rates), distances_learning_rate, zs=0, zdir='y', label='Distance to (1,1)')
    ax.set_xlabel('log10(Learning Rate)')
    ax.set_ylabel('Fixed Initial Values')
    ax.set_zlabel('Distance to (1,1)')
    plt.title("Variation of Learning Rate")

    if output_type == "show":
        plt.show()
    elif output_type == "save":
        plt.savefig("NUMERICAL_TP5_2_learning_rate_variation.png")
    plt.close(fig)
    
    # Part 2: Variation of Initial Values with Best Learning Rate
    a_inits = np.arange(-5, 6, 1)
    b_inits = np.arange(-5, 6, 1)
    distances_init = np.zeros((len(a_inits), len(b_inits)))
    
    for i, a_init in enumerate(a_inits):
        for j, b_init in enumerate(b_inits):
            if output_type == "print" and a_init == 1 and b_init == 1: # not counting 1, 1 because it would be weird also letting it for the plot because it would be weirder
                distances_init[i, j] = 10000000
                continue
            a_final, b_final = optimize_rosenbrock_numerical(a_init, b_init, learning_rate=best_learning_rate, num_iterations=2000)
            distance = distance_to_target(a_final, b_final)
            distances_init[i, j] = distance

    if output_type == "print":
        min_distance_index = np.unravel_index(np.argmin(distances_init), distances_init.shape)
        best_a_init = a_inits[min_distance_index[0]]
        best_b_init = b_inits[min_distance_index[1]]
        best_distance = distances_init[min_distance_index]
        print(f"Best Initial a: {best_a_init}, Best Initial b: {best_b_init}, Distance to Target: {best_distance}")

    # Plotting the initial values variation
    a_mesh, b_mesh = np.meshgrid(a_inits, b_inits)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(a_mesh, b_mesh, distances_init, cmap="viridis")
    ax.set_xlabel("Initial a")
    ax.set_ylabel("Initial b")
    ax.set_zlabel("Distance to (1,1)")
    plt.title("Variation of Initial Values with Best Learning Rate")
    
    if output_type == "show":
        plt.show()
    elif output_type == "save":
        plt.savefig("NUMERICAL_TP5_2_initial_values_variation.png")
    plt.close(fig)

search_best_learning_rate_and_initial_values()

# Question 1: How does the learning rate influence the convergence ?
# To have beautiful plot I decided to always have 3 variables but i would have been better to just do a gridsearch on 4D, with this method that gives wonderful plots I got with initial value (3, 5):
# Learning Rate: 0.1, Distance to Target: 3.0948500982134507e+30
# Learning Rate: 0.01, Distance to Target: 1.9146588395898e+17
# Learning Rate: 0.001, Distance to Target: 2.635192646341191e+18
# Learning Rate: 0.0001, Distance to Target: 4.213450661753289
# Learning Rate: 1e-05, Distance to Target: 4.318167514397216

# Then finding the best initial values:
# Best Learning Rate: 0.0001, Best Initial a: 5, Best Initial b: 1, Distance to Target: 0.26792544606626534

# So as we can see we need a pretty small learning rate for our values not to explode, below or equal to 0.0001


# Question 2: What problems can you encounter while using numerical gradiants based methods ?
# The main problems are the approximations, especially when doing gradiant descent you will sum many times small approximations that may lead to big errors.