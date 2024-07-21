import numpy as np
import matplotlib.pyplot as plt
import time

##########
# Cubic polynomial roots calculator
def cubic_root(coeff):
    """
    Computes the roots of a cubic equation of the form ax^3 + bx^2 + cx + d = 0.

    Parameters:
    coeff (list or numpy.ndarray): A list or array of length 4 containing the coefficients of the equation in descending order of degree.

    Returns:
    x(numpy.ndarray): A 1D array of length 3 containing the roots of the equation. The roots may be real or complex.

    Raises:
    Exception: If coeff is not a list or array of length 4, or if the coefficient of x^3 is zero.
    """
    if len(coeff) != 4:
        raise ValueError(f"The input length is {len(coeff)}. it must be 4.")
        
    a, b, c, d = coeff[0], coeff[1], coeff[2], coeff[3]
    f = ((3 * c / a) - (b**2 / a**2)) / 3
    g = (2 * b**3 / a**3 - 9 * b * c / a**2 + 27 * d / a) / 27
    h = g**2 / 4 + f**3 / 27
    
    x = []
    if h <= 0 and f != 0 and g != 0:  ## All 3 Roots Are REAL ##

        i = ((g**2 / 4) - h)**0.5
        q = np.cbrt(i)
        k = np.arccos(-g / (2 * i))
        L = q*-1
        M = np.cos(k/3)
        N = (3**0.5) * np.sin(k / 3)
        P = -(b/(3*a))
        x.append(2*q * np.cos(k / 3) - b / (3 * a))
        x.append(L * (M + N) + P)
        x.append(L * (M - N) + P)

    elif h > 0 and f != 0 and g != 0:    ## Only ONE Root is REAL ##
        R = -(g / 2) + h**0.5
        S = np.cbrt(R)
        T = -(g / 2) - h**0.5
        U = np.cbrt(T)
        x.append((S + U) - (b / (3 * a)))
        x.append((-(S + U) / 2 - (b / (3 * a)) - (S - U) * (3)**0.5 / 2j))
        x.append((-(S + U) / 2 - (b / (3 * a)) + (S - U) * (3)**0.5 / 2j))

    elif f == 0 and g == 0 and h == 0:   ## All 3 Roots Are Real and EQUAL ##
        x.append(-(d / a)**(1 / 3))
        x.append(-(d / a)**(1 / 3))
        x.append(-(d / a)**(1 / 3))

    return np.array(x)

##########
# Inverse Matrix
def inverse_matrix(matrix):
    """
    Compute the inverse of a square matrix using Gauss-Jordan elimination.

    Parameters:
    a (numpy.ndarray): The square matrix for which to find the inverse.

    Returns:
    numpy.ndarray: The inverse of the input matrix if it exists.
    
    Raises:
    ValueError: If the input matrix is not square or is not invertible.
    """
    # Check if the matrix is square
    n, m = matrix.shape
    if n != m:
        raise ValueError("The input matrix is not a square matrix.")

    # Create an augmented matrix [matrix | identity]
    augmented_matrix = np.hstack((matrix, np.eye(n)))

    # Perform row operations to get the identity matrix on the left side
    for col in range(n):
        # Scale the current row to make the diagonal element 1
        pivot = augmented_matrix[col, col]
        if pivot == 0:
            raise ValueError("Matrix is not invertible. The pivot element is zero.")

        augmented_matrix[col, :] *= (1.0 / pivot)

        # Eliminate other rows
        for row in range(n):
            if row != col:
                factor = augmented_matrix[row, col]
                augmented_matrix[row, :] -= factor * augmented_matrix[col, :]

    # Extract the right side of the augmented matrix as the inverse
    inverse = augmented_matrix[:, n:]

    return inverse

##########
# LU decomposition
def lu_decomposition(matrix):
    """
    Perform LU decomposition of a square matrix.

    Parameters:
    a (numpy.ndarray) : The square matrix to be decomposed.

    Returns:
    numpy.ndarray, numpy.ndarray: The lower triangular matrix (l) and the upper triangular  matrix (u).

    Raises:
    ValueError: If the input matrix is not square.
    """
    # Check if the matrix is square
    n, m = matrix.shape
    if n != m:
        raise ValueError("The input matrix is not a square matrix")
        
    l = np.eye(n)
    u = np.zeros((n, n))

    for j in range(n):
        for i in range(j+1):
            temp_sum = 0
            for k in range(i):
                temp_sum += l[i, k] * u[k, j]
            u[i, j] = matrix[i, j] - temp_sum


        for i in range(j+1, n):
            temp_sum = 0
            for k in range(j):
                temp_sum += l[i, k] * u[k, j]
            l[i, j] = (matrix[i, j] - temp_sum) / u[j, j]
    return l, u

# lU Decomposition Solver
def lu_solver(a, b):
    """
    Solve linear systems (Ax = b) Using LU decomposition method.

    Parameters:
    a (numpy.ndarray): The coefficients matrix.
    b (numpy.ndarray): The Right Hand Side (RHS) vector
    
    Returns:
    numpy.ndarray: The solution vector (x).
    """
    n = a.shape[0]
    z = np.zeros(n)
    x = np.zeros(n)    # x: solution vector
    l, u = lu_decomposition(a)
    
    for j in range(n):
        temp_sum = 0
        for i in range(j):
            temp_sum += l[j, i] * z[i]
        z[j] = b[j] - temp_sum
        
    for i in range(n-1,-1, -1):
        temp_sum = 0
        for j in range(i+1, n):
            temp_sum += u[i, j] * x[j]
        x[i] = (z[i] - temp_sum) / u[i, i]
    return x

##########
# Jacobian Matrix
def jacobian_matrix(equations, evaluation_point, perturbation=1e-3):
    """
    Compute the Jacobian matrix of a system of equations at a specified evaluation point.

    Parameters:
    equations (function): A function that takes a vector and returns a vector of equations to solve.
    evaluation_point (numpy.ndarray): The point at which to evaluate the Jacobian matrix.
    perturbation (float, optional): The perturbation value used for finite differences (default: 1e-6).

    Returns:
    numpy.ndarray: The Jacobian matrix at the specified evaluation point.
    """
    # Find the shape of the Jacobian matrix
    n = evaluation_point.shape[0]
    jacobian = np.zeros((n, n))
    
    # Evaluate the equations at the specified point
    eq = equations(evaluation_point)
    
    # Claculate the elements of the Jacobian matrix
    for i in range(n):
        perturbed_point = evaluation_point.copy()
        perturbed_point[i] += perturbation * abs(evaluation_point[i]) if evaluation_point[i] != 0 else perturbation
        eq_perturbed = equations(perturbed_point)
        jacobian[:, i] = (eq_perturbed - eq) / (perturbed_point[i] - evaluation_point[i])

    return jacobian

##########
# L2-Norm
def norm2(vector):
    """
    Calculate the Euclidean norm (L2 norm) of a vector.

    Parameters:
    vector (numpy.ndarray): The input vector for which to compute the norm.

    Returns:
    float: The Euclidean norm (L2 norm) of the input vector.
    """
    return (vector @ vector)**0.5

##########
# Newton's method for nonlinear systems
def newton(equations, init_guess, tol=1e-8, max_iter=100, disp=False):
    """
    Solve a system of equations using the Newton-Raphson method.

    Parameters:
    -----------
    equations (callable) :A function that takes a numpy.ndarray as input and returns a numpy.ndarray representing
    the system of nonlinear equations to be solved. The input ndarray represents the current values
    of the variables.
    init_guess (numpy.ndarray) : The initial guess for the solution of the system of equations.
    tol (float, optional): The tolerance for convergence. The iteration will stop when the L2-norm of the system of equations
    is less than this value. Default is 1e-8.
    max_iter (int, optional): The maximum number of iterations. If the convergence criterion is not met within this number of 
    iterations, a ValueError will be raised. Default is 100.
    disp (Boolian, optional): print the 2-norm of the equations for each iteration. Default is False.
    
    Returns:
    --------
    numpy.ndarray
        The solution to the system of equations.

    Raises:
    -------
    ValueError
        If the maximum number of iterations is reached without meeting the convergence criterion.
    """
    x = init_guess.copy()
    f = equations(x)
    
    # Main loop
    for k in range(max_iter):
        norm_f = norm2(f)
        
        # Display the results at each iteration 
        if disp:
            print(f"#{k}: norm = {norm_f :^30.10f}")
            
        # Check the convergence criteria
        if norm_f < tol:
            return x
        
        # Calculate new solution candidate
        jacobian = jacobian_matrix(equations, x)
        delta_x = lu_solver(jacobian, -f)
        x += delta_x
        f = equations(x)
    
    raise ValueError("Could not converge. Try new initial guess or increase the maximum number of iterations.")
    
##########
# Broyden method for nonlinear systems
def broyden(equations, init_guess, tol=1e-8, max_iter=100, disp=False):
    """
    Solve a system of nonlinear equations using the Broyden's method.

    Parameters:
    -----------
    equations (callable) :A function that takes a numpy.ndarray as input and returns a numpy.ndarray representing
    the system of nonlinear equations to be solved. The input ndarray represents the current values
    of the variables.
    init_guess (numpy.ndarray) : The initial guess for the solution of the system of equations.
    tol (float, optional): The tolerance for convergence. The iteration will stop when the L2-norm of the system of equations
    is less than this value. Default is 1e-8.
    max_iter (int, optional): The maximum number of iterations. If the convergence criterion is not met within this number of 
    iterations, a ValueError will be raised. Default is 100.
    disp (Boolian, optional): print the 2-norm of the equations for each iteration. Default is False.

    Returns:
    --------
    numpy.ndarray: The solution to the system of nonlinear equations.

    Raises:
    -------
    ValueError: If the maximum number of iterations is reached without meeting the convergence criterion.
    """
    x = init_guess.copy()
    f = equations(x)
    
    # Main loop
    for k in range(max_iter):
        norm_f = norm2(f)
        
        # Display the results at each iteration 
        if disp:
            print(f"#{k}: norm = {norm_f :^30.10f}")
        
        # Check the convergence criteria
        if norm_f < tol:
            return x
        
        # Calculate new solution candidate
        jacobian = jacobian_matrix(equations, x)
        delta_x = lu_solver(jacobian, -f)
        
        # Broyden method for step size
        s = 1
        for j in range(20):
            # Evaluate F(x)
            F = equations(x + s * delta_x)

            # Check the convergence condition
            if norm2(F) < norm_f:
                x += s * delta_x
                f = equations(x)
                break
                
            # Calculate new step size
            eta = (F @ F) / (f @ f)
            s = ((1 + 6 * eta)**0.5 - 1) / (3 * eta)
    
        else:
            raise ValueError("Could not converge. Try a new initial guess or increase the maximum number of iterations.")
            
##########
# Householder method for nonlinear systems
def householder(equations, init_guess, tol=1e-8, max_iter=100, disp=False):
    """
    Solve a system of nonlinear equations using the Householder method.

    Parameters:
    -----------
    equations (callable) :A function that takes a numpy.ndarray as input and returns a numpy.ndarray representing
    the system of nonlinear equations to be solved. The input ndarray represents the current values
    of the variables.
    init_guess (numpy.ndarray) : The initial guess for the solution of the system of equations.
    tol (float, optional): The tolerance for convergence. The iteration will stop when the L2-norm of the system of equations
    is less than this value. Default is 1e-8.
    max_iter (int, optional): The maximum number of iterations. If the convergence criterion is not met within this number of 
    iterations, a ValueError will be raised. Default is 100.
    disp (Boolian, optional): print the 2-norm of the equations for each iteration. Default is False.

    Returns:
    --------
    numpy.ndarray: The solution to the system of nonlinear equations.

    Raises:
    -------
    ValueError: If the maximum number of iterations is reached without meeting the convergence criterion.
    """
    
    x = init_guess.copy()
    f = equations(x)
    
    # Main loop
    for k in range(max_iter):
        norm_f = norm2(f)
        
        # Display the results at each iteration 
        if disp:
            print(f"#{k}: norm(f) = {norm_f}")
        
        # Check the convergence criteria
        if norm_f < tol:
            return x
        
        # Householder method
        s = 1.0
        l = 0
        for j in range(max_iter):
            if l > 2 or k == 0:
                jacobian = jacobian_matrix(equations, x)
                H = -inverse_matrix(jacobian)
                delta_x = s * H @ f
                s = 1.0
                l = 0
            
            # Evaluate F(x)
            F = equations(x + delta_x)

            # Check the convergence condition
            if norm2(F) <= norm_f:
                x_ = x + s * delta_x
                f_ = equations(x_)
                y = (f_ - f).T

                H -= ((H @ y + s * delta_x) @ (delta_x @ H)) / (delta_x @ H @ y)
                x = x_
                f = equations(x)
                s = 1.0
                l = 0
                break
            
            # Calculate new step size
            #eta = (F @ F) / (f @ f)
            #s = ((1 + 6 * eta)**0.5 - 1) / (3 * eta)
            s *= 0.7
            l += 1
    else:
        raise ValueError("Could not converge. Try a new initial guess or increase the maximum number of iterations.")


##########
# Cubic SPline
class CubicSpline:
    """
    Perform Cubic Spline interpolation on an one-dimentional dataset.

    Attributes
    ----------
    x (numpy.ndarray) : The array of x-coordinates of the data points.
    y (numpy.ndarray) : The array of y-coordinates of the data points.
    mode (int, optional) : The mode of the boundary condition. 1 for natural, 2 for clamped, 3 for not-a-knot. Default is 1.
    cubic_coeff (numpy.ndarray) : The array of coefficients of the cubic spline segments.

    Methods
    -------
    fit()
        Computes the coefficients of the cubic spline segments using a linear system of equations.
    predict(targets, draw)
        Evaluates the cubic spline at the given target points and returns the interpolated values.
    polynomial_coeff(target)
        Returns the nearest data point and the coefficients of the interpolation polynomial for that point.
    """  
    def __init__(self, x : np.ndarray, y : np.ndarray, mode : int=1):
        self.x = x
        self.y = y
        self.mode = mode
        self.cubic_coeff = self.fit()
        
    def fit(self):
        # Step 1: calculate the distance between datapoints
        n = self.x.shape[0]
        h = np.zeros(n-1) # distance between datapoints
        for i in range(0, n-1):
            h[i] = self.x[i+1] - self.x[i]
        
        # step 2: make the coefficients matrix and the right hand side vector
        rhs = np.zeros(n)            # the right hand side vector
        coeff_mat = np.zeros((n, n)) # the coefficients matrix
        for i in range(1, n-1):
            rhs[i] = 6 * ((self.y[i+1] - self.y[i]) / h[i] - (self.y[i] - self.y[i-1]) / h[i-1])
        
        for i in range(1, n-1):
            for j in range(1, n-1):
                if i == j:
                    coeff_mat[i, j] = 2 * (h[i-1] + h[i])
                elif j == i-1:
                    coeff_mat[i, j] = h[i-1]
                elif j == i+1:
                    coeff_mat[i, j] = h[i]
                    
        # Step 3: Apply the boundary conditions
        ## mode 1: Natural
        if self.mode == 1:
            coeff_mat[0, 0] = 1
            coeff_mat[-1, -1] = 1
        ## mode 2: Clamped
        if self.mode == 2:
            coeff_mat[0, 0] = 1
            coeff_mat[0, 1] = -1
            coeff_mat[-1,-1] = 1
            coeff_mat[-1, -2] = -1
        ## mode 3: Not-a-knot
        if self.mode == 3:
            coeff_mat[0, 0] = 1
            coeff_mat[0, 1] = -(1 + h[0] / h[1])
            coeff_mat[0, 2] = h[0] / h[1]
            coeff_mat[-1, -1] = 1
            coeff_mat[-1, -2] = - (1 + h[-1] / h[-2])
            coeff_mat[-1, -3] = h[-1] / h[-2]
        
        # step 4: Solve the set of linear equations
        s = lu_solver(coeff_mat, rhs)
        
        # step 5: Calculate the cubic polynomial coefficients
        cubic_coeff = np.zeros((n, 4))
        for i in range(n-1):
            cubic_coeff[i, 0] = (s[i+1] - s[i]) / (6 * h[i])
            cubic_coeff[i, 1] = s[i] / 2
            cubic_coeff[i, 2] = (self.y[i+1] - self.y[i]) / h[i] - (2 * h[i] * s[i] + h[i] * s[i+1]) / 6
            cubic_coeff[i, 3] = self.y[i]
    
        return cubic_coeff
    
    def predict(self, target):
        """
        Calculate the predicted value for target points.

        Parameters:
        targets (numpy.ndarray): The points that the value of them should be predicted.        
        
        Returns:
        numpy.ndarray: The vector of predicted values for target points.

        Raises:
        ValueError: If the input point is not within the range of initial dataset.
        """
        # Exceptions:
        if target < min(self.x) or target > max(self.x):
            raise ValueError(f"""The entered point is not in the range of initial dataset.
            target point must be within ({min(self.x)}, {max(self.x)})""")
            

        # step 1: find the nearest data point
        for i in range(self.x.shape[0]-1):
            if target >= self.x[i] and target < self.x[i+1]:
                index = i
            elif target == self.x[-1]:
                index = len(self.x) - 2

        # step 2: predict the value for target
        prediction = (
            self.cubic_coeff[index, 0] * (target - self.x[index])**3
          + self.cubic_coeff[index, 1] * (target - self.x[index])**2
          + self.cubic_coeff[index, 2] * (target - self.x[index])
          + self.cubic_coeff[index, 3])
            
        return prediction
    
    def polynomial_coeff(self, target):
        """
        Returns the  nearest data point and the coefficients of the cubic polynomial to interpolate the target point.
            y = a * (x-x0)**3 + b * (x - x0)**2 + c * (x - x0) + d
            
        Parameters:
        target (float): the point at which the polynomial coefficient are required.
        
        Returns:
        (x0, numpy.ndarray): Nearest data point to the target, the coefficients of the cubic polynomial [a, b, c, d].
        
        Raises:
        ValueError: If any of the input points is not within the range of initial dataset.
        """
        # Exceptions:
        if target < self.x[0] or target > self.x[-1]:
            raise ValueError(f"""The entered point is not in the range of initial dataset.
            targets must be within ({self.x[0]}, {self.x[-1]})""")
        
        # Find the nearest data point
        for i in range(self.x.shape[0]-1):
            if target >= self.x[i] and target < self.x[i+1]:
                index = i
            elif target == self.x[-1]:
                index = len(self.x) - 2
        return self.x[index], self.cubic_coeff[index, :]
    
