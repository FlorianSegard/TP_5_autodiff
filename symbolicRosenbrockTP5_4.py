from sympy import diff, symbols
import matplotlib.pyplot as plt
import numpy as np
from sympy import sqrt


def rosenbrock_symbolic_gradient(expr, a, b):
    return diff(expr, a), diff(expr, b)
    

a, b = symbols('a b')
expr = (1 - a)**2 + 100 *(b - a**2)**2

diff_a, diff_b = rosenbrock_symbolic_gradient(expr=expr, a=a, b=b)

print(rosenbrock_symbolic_gradient(expr=expr, a=a, b=b))


def optimize_rosenbrock_symbolic(a_init, b_init, diff_a, diff_b, learning_rate=0.001, num_iterations=2000):
    args = [a_init, b_init]
    for _ in range(num_iterations):
        current_args_descent = [diff_a.subs({a: args[0], b: args[1]}), diff_b.subs({a: args[0], b: args[1]})]
        if (args[0] > 1000 or args[1] > 1000): # Because sometime the value become bigger than 10e100000000000000000000000000000000000000000000000000000000000000000000...
            
            print("lr: ", learning_rate)
            print("breaked")
            break

        args[0] -= learning_rate * current_args_descent[0]
        args[1] -= learning_rate * current_args_descent[1]
    return args[0], args[1]


# Searching best learning rate and then best initial values and ploting them in png files.
# You can chose print to print in terminal, show to have plt.show for interactive or save for saving in .png file
def symbolic_search_best_learning_rate_and_initial_values(output_type="save"):
    # Parameters
    learning_rates = [0.1 * 10**-i for i in range(5)]
    distances_learning_rate = []
    a_init, b_init = 3, 5
    
    # Define the distance function
    def distance_squared_to_target(a, b):
        return (a - 1) ** 2 + (b - 1) ** 2
    
    # Part 1: Variation of Learning Rate
    for lr in learning_rates:
        a_final, b_final = optimize_rosenbrock_symbolic(a_init, b_init, diff_a, diff_b, learning_rate=lr, num_iterations=2000)
        distance = distance_squared_to_target(a_final, b_final)
        distances_learning_rate.append(distance)
        if output_type == "print":
            print(f"Learning Rate: {lr}, Distance to Target: {distance}")
    print(distances_learning_rate)
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
        plt.savefig("SYMBOLIC_TP5_4_learning_rate_variation.png")
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
            a_final, b_final = optimize_rosenbrock_symbolic(a_init, b_init, diff_a, diff_b, learning_rate=best_learning_rate, num_iterations=2000)
            distance = distance_squared_to_target(a_final, b_final)
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
        plt.savefig("SYMBOLIC_TP5_4_initial_values_variation.png")
    plt.close(fig)

symbolic_search_best_learning_rate_and_initial_values()



# Question 1: Compare the efficacity of the optimization methods using both symbolic and numerical gradients.
# For the creation of the 3D plot is like 10 times slower but it seems more precise in terms of value.

# Question 2: How does the calculus precision of the symbolic gradients affect optimization ?
# The precision is very high and the gradients can become very very big, that's why I had to put the if.