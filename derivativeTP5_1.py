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


# QUESTION: Quelle est l’influence du choix du pas h sur la précision de l’approximation ?
# h=1 semble être un cas particulier mais sinon le plus précis est 0.01

# QUESTION: Comparez cette méthode avec les différences finies avant et arrière. Quand une méthode est-elle préférable ?
# Lorsque h est petit grand et que l'on a peu de précision il vaut mieux utiliser la différence finie centrale de plus il peut être préférable d'utiliser la backward ou la forward dans le cas où la fonction n'est dérivable qu'à droite ou qu'à gauche.
