from TP5_1 import compute_mean_diff, finite_derivative, finite_derivative_forward, finite_derivative_backward
from sympy import diff, symbols, lambdify
import numpy as np

x = symbols('x')
f = x**3

f_deriv = diff(f, x)

f_func = lambdify(x, f, 'numpy')
f_deriv_func = lambdify(x, f_deriv, 'numpy')



if __name__ == "__main__":


    array = np.arange(0, 151)

    for i in range(5):

        print(i)
        print("Centrale: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative, h=10 ** (-i), f=f_func, f_deriv=f_deriv_func))
        print("Forward: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative_forward, h=10 ** (-i), f=f_func, f_deriv=f_deriv_func))
        print("Backward: ", compute_mean_diff(array_numbers=array, derivative=finite_derivative_backward, h=10 ** (-i), f=f_func, f_deriv=f_deriv_func))
        print("--------------------")




# Question 1: What are the limitations of the finite derivate in terms of precision and performance ?
# In term of precision the finite_derivate is pretty good but the two others are very far from the symbolic derivative.

# Question 2: When would we prefer using symbolic derivates instead of finite derivates ?
# Symbolic derivates are better if we want to be precise in fact if you do many iterations the smalls errors can multipliate.
# In Deep Learning the basic case would be grandient descent with many iterations.
