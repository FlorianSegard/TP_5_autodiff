from sympy import diff, symbols

def symbolic_derivative(expr, var):
    return diff(expr, var)

x = symbols('x')
expr = x ** 2
real_deriv = 2 * x
expr_deriv = symbolic_derivative(expr, x)
print(expr_deriv)

assert real_deriv == expr_deriv

# Question 1: Discuss about the downside and upside of symbolic derivative compared to numerical derivative.
# Upsides: 
# - Symbolic derivative do not come with many small errors that might lead to big ones especially with how we use it on Deep Learning with gradiant descent.
# - Symbolic derivative might be slower for big formulas but you can calculate once and stock the value of the derivative to reuse it after not having to calculate it everytime like with numerical derivative. So on very big amount of epochs it may compensate.

# Downsides:
# - Symbolic derivative can create enormous calculation graph for very complex expressions which can be problematic.

# Question 2: How does SymPy manage complex expressions ?
# - SymPy creates a calculation graph.
# - It simplifies expressions.
# - It uses many different techniques to optimize such as chain rule or product derivative.