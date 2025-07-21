"""
Find the mathematical function skeleton that represents E. Coli bacterial growth rate, given data on population density, substrate concentration, temperature, and pH level. 
You can define scipy.optimize.minimize using equation.config to set:
- x0: Initial guess (e.g., [1.0, 0.5, ...])
- method: Optimization algorithm (e.g., 'BFGS', 'L-BFGS-B', 'Powell', 'SLSQP')
- options: Optimizer settings (e.g., {'maxiter': 300, 'ftol': 1e-8})
- bounds: Variable bounds as (min, max) tuples, (None, None) for no bounds
- constraints: List of equality or inequality constraints
"""


import numpy as np

#Initialize parameters
MAX_NPARAMS = 10
PRAMS_INIT = [1.0]*MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations."""
    
    # Load data observations
    inputs, outputs = data['inputs'], data['outputs']
    b, s, temp, pH = inputs[:,0], inputs[:,1], inputs[:,2], inputs[:,3]
    
    # Optimize parameters based on data
    from scipy.optimize import minimize
    def loss(params):
        y_pred = equation(b, s, temp, pH, params)
        return np.mean((y_pred - outputs) ** 2)

    loss_partial = lambda params: loss(params)

    if hasattr(equation, 'config') and 'x0' in equation.config:
        result = minimize(loss_partial, **equation.config)
    else:
        result = minimize(loss_partial, [1.0]*MAX_NPARAMS, method='BFGS')
    
    # Return evaluation score
    optimized_params = result.x
    loss = result.fun

    if np.isnan(loss) or np.isinf(loss):
        return None
    else:
        return -loss



@equation.evolve
def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ Mathematical function for bacterial growth rate

    Args:
        b: A numpy array representing observations of population density of the bacterial species.
        s: A numpy array representing observations of substrate concentration.
        temp: A numpy array representing observations of temperature.
        pH: A numpy array representing observations of pH level.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing bacterial growth rate as the result of applying the mathematical function to the inputs.
    """
    equation.config = {
        'x0': [1.0] * 5,
        'method': 'BFGS',
        'bounds': [(None, None)] * 5,
        'constraints': []
    }
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]