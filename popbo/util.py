import numpy as np
import scipy as sp
import safeopt
import GPy
import os
import sys
import torch


sys.path.append('./pbosgp/preferentialBO/test_functions/')
try:
    from test_functions.test_functions import *
except:
    from test_functions import *

# Function to generate and sample Gaussian Process (GP) functions
def try_sample_gp_func(random_sample, num_knots, problem_dim, bounds, config,
                       kernel, numerical_epsilon, noise_var, get_zero=False):
    # If random sampling is enabled, generate random sample points
    if random_sample:
        sample_list = []
        # For each sample, add one point (problem_dim)
        for i in range(num_knots):
            loc = np.random.uniform(0.0, 1.0, problem_dim)
            x = np.array([bounds[i][0] +
                         (bounds[i][1] - bounds[i][0]) * loc[i]
                          for i in range(problem_dim)]
                         )
            sample_list.append(x)
        knot_x = np.array(sample_list)
    # Generate linearly spaced combinations if not random
    else:
        knot_x = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )
    # Configuration and calculation of covariance matrix
    config['knot_x'] = knot_x
    knot_cov = kernel.K(config['knot_x']) + \
        np.eye(knot_x.shape[0]) * numerical_epsilon
    knot_cov_cho = sp.linalg.cho_factor(knot_cov)
    # Define function based on get_zero flag
    if get_zero:
        fun = lambda x: np.zeros(shape=(x.shape[0], 1))
    else:
        fun = safeopt.sample_gp_function(
            kernel,
            config['bounds'],
            noise_var,
            config['discretize_num_list']
        )
    # Calculate function values and alpha
    knot_y = fun(knot_x)
    print(knot_y)
    alpha = sp.linalg.cho_solve(knot_cov_cho, knot_y)
    # Update config with calculated values
    config['knot_cov'] = knot_cov
    config['knot_cov_cho'] = knot_cov_cho
    config['knot_y'] = knot_y
    config['knot_min'] = np.min(knot_y)
    config['alpha'] = alpha
    config['func_norm'] = np.sqrt((knot_y.T @ alpha)[0, 0])
    return config

# Function to configure the Bayesian optimization problem
def get_config(problem_name, problem_dim=None, GP_kernel=None, discret_num=3,
               Gamma=None, num_knots=10, random_sample=True, samples_max=20):
    """
    Configure the GP problem.
    Inputs:
    - problem_name: Name of the GP problem.
    - problem_dim: Dimensionality of the problem.
    - GP_kernel: Type of GP kernel to use.
    - discret_num: Number of discretization points.
    - Gamma: Parameter for optimization.
    - num_knots: Number of sample points (knots).
    - random_sample: Flag to use random sampling.
    - samples_max: Maximum number of samples to try.
    Output: Configuration dictionary for the problem.
    """
    numerical_epsilon = 1e-6
    config = dict()
    config['problem_name'] = problem_name
    config['numerical_epsilon'] = numerical_epsilon

    if problem_name == 'GP_sample_func':
        # Default problem dimension and kernel if not specified
        if problem_dim is None:
            problem_dim = 2
        if GP_kernel is None:
            GP_kernel = 'Gaussian'

        if problem_dim == 1:
            config['beta_func'] = lambda t: 0.1 * np.sqrt(t+1)

        # Set problem dimension and discretization
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [discret_num] * problem_dim
        bounds = [[0, 10]] * problem_dim # use [0, 1] sequence.
        config['bounds'] = bounds
        # Kernel configuration based on specified GP_kernel
        kernel_scale = 3.0
        # Gaussian (RBF) kernel
        if GP_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=1.*kernel_scale**2,
                                  lengthscale=1.0, ARD=True)
        # Polynomial kernel
        if GP_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0 * kernel_scale**2,
                                   scale=1.0,
                                   order=1)
        # Matern 5/2 kernel
        if GP_kernel == 'Matern52':
            kernel = GPy.kern.Matern52(
                    input_dim=len(config['bounds']),
                    variance=1.0 * kernel_scale**2,
                    lengthscale=1.0,
                    ARD=True
                )

        noise_var = 0.0
        config['noise_var'] = noise_var

        num_sample_try = 0  # record the number of knots tried.
        # Sample from GP and adjust configuration based on Gamma parameter
        while True:
            config = try_sample_gp_func(random_sample, num_knots, problem_dim,
                                        bounds, config, kernel,
                                        numerical_epsilon, noise_var)
            num_sample_try += 1
            if num_sample_try > samples_max:
                num_knots = max(num_knots-1, 1)  # if the number

            if Gamma is None:
                config['Gamma'] = config['func_norm'] * (1 + 0.1)
                break
            else:
                if config['func_norm'] <= Gamma:
                    config['Gamma'] = Gamma
                    break
        # Define objective function using sampled GP
        knot_x = config['knot_x']
        alpha = config['alpha']

        def obj_f(x):
            x = np.atleast_2d(x)
            y = kernel.K(x, knot_x).dot(alpha)
            y = np.squeeze(y)
            return y
        if problem_dim == 1:
            config['x_grid'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [30] * problem_dim
        )
        else:
            config['x_grid'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [10] * problem_dim
        )


        config['norm_bound'] = config['Gamma']
        config['obj'] = obj_f
        config['kernel'] = kernel
        obj_val_list = [obj_f(x) for x in config['x_grid']]
        config['opt_val'] = max(obj_val_list)

    if problem_name == 'thermal_1d':
        # optimizing temperature 1d
        if problem_dim is None:
            problem_dim = 1
        if GP_kernel is None:
            GP_kernel = 'Gaussian'


        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [discret_num] * problem_dim
        bounds = [[20, 30]]
        config['bounds'] = bounds
        avg_bounds_len = np.mean([bound[1] - bound[0] for bound in bounds])
        kernel_scale = 2.0
        kernel_lengthscale = avg_bounds_len / 2.0
        config['x_grid'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [30] * problem_dim
        )


        # Define Kernel
        if GP_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']),
                                  variance=1.*kernel_scale**2,
                                  lengthscale=kernel_lengthscale, ARD=True)
        if GP_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0 * kernel_scale**2,
                                   scale=kernel_lengthscale,
                                   order=1)
        if GP_kernel == 'Matern52':
            kernel = GPy.kern.Matern52(
                    input_dim=len(config['bounds']),
                    variance=1.0 * kernel_scale**2,
                    lengthscale=kernel_lengthscale,
                    ARD=True
                )

        noise_var = 0.0
        config['noise_var'] = noise_var

        config['Gamma'] = 3


        def obj_f(x):
            return _PPD(x)
        config['norm_bound'] = config['Gamma']
        config['obj'] = obj_f
        config['kernel'] = kernel
        obj_val_list = [obj_f(x) for x in config['x_grid']]
        # print(config['x_grid'])
        # print(obj_val_list)
        config['opt_val'] = max(obj_val_list)

    if 'test_func_' in problem_name:
        test_func_name_to_std = {
            'Forrester': 1.0,
            'Branin': 1.0,
            'CurrinExp': 1.0
        }
        test_func_name_to_div = {
            'Styblinski_tang': 3,
            'Bukin': 4,
            'Cross_in_tray': 4,
            'Eggholder': 3,
            'Holder_table': 3,
            'Langerman': 3,
            'Levy13': 3,
            'Shubert': 3
        }

        test_func_name_to_beta_0 = {
            'Forrester': 1,
            'Branin': 10,
            'CurrinExp': 10,
            'Styblinski_tang_': 1,
            'Bukin': 3,
            'Eggholder': 0.5,
            'Langerman': 1.0,
            'Shubert': 0.5
        }
        test_func_name = problem_name[10:]
        test_func_std = 1.0 # test_func_name_to_std[test_func_name]
        test_func = eval(test_func_name)()
        problem_dim = test_func.d
        if problem_dim is None:
            problem_dim = test_func.d
        if GP_kernel is None:
            GP_kernel = 'Matern52' #'Gaussian'

        if test_func_name in test_func_name_to_beta_0.keys():
            beta_0 = test_func_name_to_beta_0[test_func_name]
            #2 #10 # 3
        else:
            beta_0 = 1.0


        config['beta_func'] = lambda t: beta_0 * np.sqrt(t+1)
        test_func_name_to_norm_bound = {
            'Branin': 10,
            'CurrinExp': 5,
            'Styblinski_tang_': 5,
            'Bukin': 4,
            'Langerman': 3,
            'Eggholder': 3
        }
        if test_func_name in test_func_name_to_norm_bound.keys():
            config['norm_bound'] = test_func_name_to_norm_bound[test_func_name]
            #2 #10 # 3
        else:
            config['norm_bound'] = 6

        config['var_dim'] = problem_dim
        discret_num = 20
        config['discretize_num_list'] = [discret_num] * problem_dim
        eps_bound = 1e-6
        bounds = [
            [test_func.bounds[0][k]+eps_bound, test_func.bounds[1][k]-eps_bound]
                  for k in range(problem_dim)]  # use [0, 1] sequence.
        config['bounds'] = bounds
        avg_bounds_len = np.mean([bound[1] - bound[0] for bound in bounds])


        test_func_name_to_lenscale = {
            'Branin': [10, 10],
            'CurrinExp': [0.1, 0.1],
            'Styblinski_tang_': [1, 1],
            'Bukin': [1, 1],
            'Eggholder': [80, 80],
            'Holder_table': [2, 2],
            'Langerman': [2, 2],
            'Levy13': [2, 2],
            'Shubert': [0.8, 0.8]
        }
        if test_func_name in test_func_name_to_lenscale.keys():
            kernel_lengthscale = test_func_name_to_lenscale[test_func_name]
        else:
            kernel_lengthscale = avg_bounds_len / 2.0
        # Define Kernel
        if GP_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']),
                                  variance=test_func_std**2,
                                  lengthscale=kernel_lengthscale, ARD=True)
        if GP_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=test_func_std**2,
                                   scale=kernel_lengthscale,
                                   order=1)
        if GP_kernel == 'Matern52':
            kernel = GPy.kern.Matern52(
                    input_dim=len(config['bounds']),
                    variance=test_func_std**2,
                    lengthscale=kernel_lengthscale,
                    ARD=True
                )

        noise_var = 0.0
        config['noise_var'] = noise_var


        def orig_obj_f(x):
            y = test_func.values(x)
            if torch.numel(torch.tensor([y])) == 1:
                return np.array(y[0,0])
            else:
                return y
        config['x_grid'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )
        y_val_grid = [orig_obj_f(x) for x in config['x_grid']]
        orig_y_mean = np.mean(y_val_grid)
        orig_y_std = np.std(y_val_grid)
        config['orig_y_mean'] = orig_y_mean
        config['orig_y_std'] = orig_y_std
        def obj_f(x):
            orig_y = test_func.values(x)
            if torch.numel(torch.tensor([orig_y])) == 1:
                orig_y = np.array(orig_y[0,0])
            y = (orig_y - orig_y_mean) / orig_y_std
            return y

        config['obj'] = obj_f
        config['kernel'] = kernel
        obj_val_list = [obj_f(x) for x in config['x_grid']]
        config['opt_val'] = max(obj_val_list)

    return config


if __name__ == '__main__':
    a = get_config('GP_sample_func', Gamma=10.0, random_sample=True,
                   num_knots=8)
    print(a)
