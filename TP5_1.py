import numpy as np

def finite_derivative(f, a, h=0.01):
    return (f(a + h) - f(a - h)) / (2 * h)

def finite_derivative_forward(f, a, h=0.01):
    return (f(a + h) - f(a)) / h

def finite_derivative_backward(f, a, h=0.01):
    return (f(a) - f(a - h)) / h


def f(x):
    return x**2

def f_deriv(x):
    return 2*x

def compute_mean_diff(array_numbers, derivative, h, f, f_deriv):
    return derivative(f, array_numbers, h).mean() - f_deriv(array_numbers).mean()


if __name__ == "__main__":

    array = np.arange(0, 151)

    for i in range(5):

        print(i)
        print("Centrale: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative, h=10 ** (-i), f=f, f_deriv=f_deriv))
        print("Forward: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative_forward, h=10 ** (-i), f=f, f_deriv=f_deriv))
        print("Backward: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative_backward, h=10 ** (-i), f=f, f_deriv=f_deriv))
        print("--------------------")


# Question 1: What is the influence of h on the approximation precision ?
# h=1 seems like a special case otherwise the best case is h=0.01

# Question 2: Compare this method with the backward and forward. Which one is better ?
# It seems like we would always want to take the central one, but in the case where the function do not have a derivate on the left or the right we might want to use one of the others.
