"""
Implement POP-BO.
"""
import numpy as np
import casadi
import time


def transform_to_2d(an_array):
    if type(an_array) == list:
        an_array = np.array(an_array)
    if an_array.ndim == 1:
        an_array = np.expand_dims(an_array, axis=0)
    elif an_array.ndim == 0:
        an_array = np.atleast_2d(an_array)
    return an_array


class POPBO():

    def __init__(self, config, do_presolve=True, solver='ipopt',
                 y_lb=-3, y_ub=3):

        if 'beta_func' in config.keys():
            self.beta_func = config['beta_func']
        else:
            self.beta_func = lambda t: 0.1 * np.sqrt(t+1)

        self.norm_bound = config['norm_bound']

        self.config = config
        self.x_grid = config['x_grid']
        num_x, dim_x = config['x_grid'].shape
        self.num_x = num_x
        self.dim_x = dim_x
        self.kernel = config['kernel']
        self.orig_len_scale = self.kernel.lengthscale
        print(self.orig_len_scale)
        self.x_prime_list = []
        self.x_list = []
        self.one_list = []
        self.width_list = []

        self.estimated_best_x_list = []
        self.estimated_best_x_val_list = []
        self.num_data = 0

        # initialize maximum loglikelihood to a small value
        self.max_LL = - 1e10
        self.max_LL_y = None
        self.var_lb = y_lb
        self.var_ub = y_ub
        self.numerical_epsilon = 1e-6
        self.do_presolve = do_presolve
        self.solver = solver

        # \in ['co_opt', 'grid', 'zero_order', 'test']
        self.query_method = 'co_opt'
        # \in ['robust_adv', 'grid_max_mle', 'max_mle', 'min_uncer']
        self.best_x_estim_method = 'max_mle'
        self.update_len_scale_period = 28

    def updated_estim_best_x(self):
        opt_val = self.config['opt_val']
        if self.num_data > 0 and self.best_x_estim_method == 'min_uncer':
            estim_best_x_id = np.argmin(self.width_list)
            estim_best_x = self.x_list[estim_best_x_id]
            estim_best_x_val = self.config['obj'](estim_best_x)
            print(f'Estimated best x: {estim_best_x}, ' +
                  f' with regret {opt_val - estim_best_x_val}.')
            self.estimated_best_x_list.append(estim_best_x)
            self.estimated_best_x_val_list.append(estim_best_x_val)

        if self.num_data > 0 and self.best_x_estim_method == 'robust_adv':
            input_x = [self.x_prime_list[0]] + self.x_list
            assert len(input_x) == len(self.max_LL_y)
            pick_num = min(len(input_x), 3)
            xy_ = list(zip(input_x,self.max_LL_y))
            sorted_xy_ = sorted(xy_, key=lambda u: u[1], reverse=True)
            cand_x = [sorted_xy_[k][0] for k in range(pick_num)]
            cand_x_lower_y_list = []
            for x in cand_x:
                y = self.get_lower_bound(x, beta=1.0)
                cand_x_lower_y_list.append(y)

            estim_best_x_id = np.argmax(cand_x_lower_y_list)
            estim_best_x = cand_x[estim_best_x_id]
            estim_best_x_val = self.config['obj'](estim_best_x)
            print(f'Estimated best x: {estim_best_x}, ' +
                  f' with regret {opt_val - estim_best_x_val}.')
            self.estimated_best_x_list.append(estim_best_x)
            self.estimated_best_x_val_list.append(estim_best_x_val)

        if self.num_data > 0 and self.best_x_estim_method == 'max_mle':
            input_x = [self.x_prime_list[0]] + self.x_list
            assert len(input_x) == len(self.max_LL_y)
            estim_best_x_id = np.argmax(self.max_LL_y)
            estim_best_x = input_x[estim_best_x_id]
            estim_best_x_val = self.config['obj'](estim_best_x)
            print(f'Estimated best x: {estim_best_x}, ' +
                  f' with regret {opt_val - estim_best_x_val}.')
            self.estimated_best_x_list.append(estim_best_x)
            self.estimated_best_x_val_list.append(estim_best_x_val)

        if self.num_data > 0 and self.best_x_estim_method == 'grid_max_mle':
            grid_y = []
            for x in self.x_grid:
                y = self.get_maxLL_min_norm_inter(x)
                grid_y.append(y)

            estim_best_x_id = np.argmax(grid_y)
            estim_best_x = self.x_grid[estim_best_x_id]
            estim_best_x_val = self.config['obj'](estim_best_x)
            print(f'Estimated best x: {estim_best_x}, ' +
                  f' with regret {opt_val - estim_best_x_val}.')
            self.estimated_best_x_list.append(estim_best_x)
            self.estimated_best_x_val_list.append(estim_best_x_val)

    def add_new_data(self, x_prime, x, one):
        width = self.get_max_adv(x, x_prime) + self.get_max_adv(x_prime, x)
        self.updated_estim_best_x()
        self.x_prime_list.append(x_prime)
        self.x_list.append(x)
        self.one_list.append(one)
        self.width_list.append(width)
        # the minimum width duel should be the final point to report
        self.num_data = self.num_data + 1

        _ = self.update_MLL()

    def get_symb_K(self, input_x_var, len_scale_var):
        # calculate and set objective
        config = self.config
        x_dim = config['var_dim']
        kernel = config['kernel']
        var = kernel.variance[0]
        len_scale = len_scale_var
        points_num, x_dim = np.array(input_x_var).shape
        divisor_epsi = 1e-20

        diff_x_scaled_0 = (
            np.expand_dims(input_x_var[0],axis=1) - input_x_var.T) / len_scale
        diff_x_scaled_inner_prod = \
            diff_x_scaled_0[:, 0].T@diff_x_scaled_0[:, 0] \
            + divisor_epsi

        for j in range(1, points_num):
            diff_x_scaled_inner_prod_ele = \
                diff_x_scaled_0[:, j].T@diff_x_scaled_0[:,j] + divisor_epsi
            diff_x_scaled_inner_prod = casadi.horzcat(
                diff_x_scaled_inner_prod,
                diff_x_scaled_inner_prod_ele
            )

        for i in range(1, points_num):
            diff_x_scaled = (
                np.expand_dims(input_x_var[i],axis=1) - input_x_var.T
            ) / len_scale
            diff_x_scaled_inner_prod_row = \
            diff_x_scaled[:, 0].T@diff_x_scaled[:, 0] \
            + divisor_epsi

            for j in range(1, points_num):
                diff_x_scaled_inner_prod_ele = \
                    diff_x_scaled[:, j].T@diff_x_scaled[:,j] + divisor_epsi
                diff_x_scaled_inner_prod_row = casadi.horzcat(
                    diff_x_scaled_inner_prod_row,
                    diff_x_scaled_inner_prod_ele
                )

            diff_x_scaled_inner_prod = casadi.vertcat(
                diff_x_scaled_inner_prod,
                diff_x_scaled_inner_prod_row
            )

        if kernel.name == 'rbf':
            K = var * casadi.exp(- 0.5 * diff_x_scaled_inner_prod)
        elif kernel.name == 'Mat52':
            r = casadi.sqrt(diff_x_scaled_inner_prod)
            K = var * (1 + np.sqrt(5.) * r+ 5./3 * r**2) * \
                casadi.exp(-np.sqrt(5.) * r)
        return K

    def get_symb_kt(self, param_input_x, var_x):
        # calculate and set objective
        config = self.config
        x_dim = config['var_dim']
        kernel = config['kernel']
        var = kernel.variance[0]
        if x_dim == 1:
            len_scale = kernel.lengthscale[0]
        else:
            len_scale = np.array(
                [kernel.lengthscale[k] for k in range(x_dim)]
            )
        points_num, x_dim = np.array(param_input_x).shape
        divisor_epsi = 0
        diff_x_scaled = (var_x - param_input_x.T) / len_scale
        diff_x_scaled_inner_prod = diff_x_scaled[:, 0].T@diff_x_scaled[:, 0] \
            + divisor_epsi
        #print(diff_x_scaled_inner_prod)
        for i in range(1, points_num):
            diff_x_scaled_inner_prod = casadi.vertcat(
                diff_x_scaled_inner_prod,
                diff_x_scaled[:, i].T@diff_x_scaled[:, i] + divisor_epsi
            )

        if kernel.name == 'rbf':
            k_input_x_var_x = var * casadi.exp(- 0.5 * diff_x_scaled_inner_prod)
        elif kernel.name == 'Mat52':
            r = casadi.sqrt(diff_x_scaled_inner_prod)
            k_input_x_var_x = var * (1 + np.sqrt(5.) * r+ 5./3 * r**2) * \
                casadi.exp(-np.sqrt(5.) * r)
        return k_input_x_var_x

    def query_new_point_co_opt(self, x_prime):
        x_grid = self.config['x_grid']
        len_x_grid = len(x_grid)
        best_x = None
        best_opt_improve = - 1e30

        for k in range(10):
            x0_id = min(int(np.random.rand()*len_x_grid), len_x_grid-1)
            x0 = x_grid[x0_id]
            new_x, new_imp = self.query_new_point_co_opt_with_xinit(
                x_prime, x0
            )
            #print(f'{k}-th trial for co_opt.')
            #print(x0, new_x)
            #print(new_x, new_imp)
            if new_imp > best_opt_improve:
                best_x = new_x
                best_opt_improve = new_imp
        if best_x.ndim == 0:
            best_x = np.expand_dims(best_x, axis=0)
        real_imp = self.get_max_adv(best_x, x_prime)
        print(
            f'{best_x} with opt improvement {best_opt_improve}' +
            f', real {real_imp}')
        if best_x.ndim == 0:
            best_x = [best_x.tolist()]
        else:
            best_x = best_x.tolist()
        return best_x

    def query_new_point_co_opt_with_xinit(self, x_prime, x0):
        # first update the maximum log likelihood

        # lists to store diferent converged solutions
        converge_sol_list = []
        converge_val_list = []
        converge_x_list = []
        # construct some parameters of the optimization
        config = self.config
        x_dim = config['var_dim']
        kernel = config['kernel']
        bounds = config['bounds']
        var = kernel.variance[0]
        len_scale = kernel.lengthscale[0]

        # compute the optimistic advantage of f(x) - f(x_prime)

        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()

            var_x = opti.variable(x_dim)
            if self.num_data > 0:
                input_x = [self.x_prime_list[0]] + self.x_list
            else:
                input_x = [x_prime]
            input_x_arr = np.array(input_x)
            if input_x_arr.ndim == 1:
                input_x_arr = np.expand_dims(input_x_arr, axis=1)
            one_arr = np.array(self.one_list)

            input_cov = kernel.K(input_x_arr) + \
                np.eye(self.num_data + 1) * self.numerical_epsilon
            input_cov_inv = np.linalg.inv(input_cov)

            var_y = opti.variable(self.num_data+2)   # variable to be optimized
            if self.num_data > 0:
                min_norm_y = self.get_maxLL_min_norm_inter(x0)
                var_y_start = np.append(self.max_LL_y, min_norm_y)
                opti.set_initial(var_y, var_y_start)
            opti.set_initial(var_x, x0)
            for k in range(self.num_data + 2):
                opti.subject_to(
                    var_y[k] <= self.var_ub
                )
                opti.subject_to(
                    var_y[k] >= self.var_lb
                )
            for k in range(x_dim):
                opti.subject_to(
                    var_x[k] <= bounds[k][1]
                )
                opti.subject_to(
                    var_x[k] >= bounds[k][0]
                )
            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                           casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            opti_improve = var_y[-1]-var_y[-2]
            opti.minimize(-opti_improve)

            kt = self.get_symb_kt(input_x_arr, var_x)
            kxx = kernel.K(np.array([[0 for _ in range(x_dim)]]))[0,0]
            # assuming the kernel is stationary
            Px = casadi.sqrt(casadi.fmax(kxx-kt.T@input_cov_inv@kt, 1e-10)) + 1e-15
            inv_1 = casadi.MX.zeros(
                self.num_data+2, self.num_data+2
            )
            inv_1[:self.num_data+1, :self.num_data+1] = input_cov_inv

            inv_2_sqrt = casadi.MX.zeros(
                self.num_data+2, 1)
            Kinv_k = input_cov_inv@kt
            inv_2_sqrt[:self.num_data+1, :] = Kinv_k
            inv_2_sqrt[-1, :] = -1
            inv_2 = (inv_2_sqrt@inv_2_sqrt.T)/(Px**2)

            opti.subject_to(
                var_y.T@(inv_1+inv_2)@var_y <= self.norm_bound**2
            )


            if self.num_data > 0:
                opti.subject_to(
                    LL >= self.max_LL - self.beta_func(self.num_data)
                )

            # set solver: 'ipopt'
            opti.solver('ipopt', dict(print_time=False), dict(print_level=False,
                                                          max_iter=100))
            try_solve_limit = 1
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(opti_improve))
                    converge_x_list.append(sol.value(var_x))
                    #print(opti.debug.show_infeasibilities())
                except Exception as e:
                    print_err = False
                    if print_err:
                        print(e, ' in optimization_problem.')
                        print(
                        'Optimistic improvement is ' +
                        f'{opti.debug.value(opti_improve)}'
                        )
                        print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(opti_improve))
                    converge_x_list.append(opti.debug.value(var_x))
                    #print(opti.debug.show_infeasibilities())

            best_imp = np.max(converge_val_list)
            best_conv_sol_id = np.argmax(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])
            best_x = np.squeeze(converge_x_list[best_conv_sol_id])

        return best_x, best_imp

    def query_new_point(self, x_prime):
        # first update the maximum log likelihood
        best_x = None
        best_opt_improve = - 1e30
        if self.query_method == 'grid':
            for x in self.x_grid:
                if type(x) == np.ndarray:
                    x = x.tolist()
                time_start = time.time()
                #print(f'get_max_adv start!')
                opt_improve = self.get_max_adv(x, x_prime)
                time_end = time.time()
                #print(f'get_max_adv for {self.num_data} time:' +
                #      f'{time_end-time_start}s!')
                # print(x, x_prime, opt_improve)
                if opt_improve > best_opt_improve:
                    best_opt_improve = opt_improve
                    best_x = x
            if type(best_x) is not list:
                best_x = [best_x]
            print(f'{best_x} with opt imp {best_opt_improve}')
        elif self.query_method == 'co_opt':

            best_x = self.query_new_point_co_opt(x_prime)
        elif self.query_method == 'test':
            for x in self.x_grid:
                if type(x) == np.ndarray:
                    x = x.tolist()
                time_start = time.time()
                #print(f'get_max_adv start!')
                opt_improve = self.get_max_adv(x, x_prime)
                time_end = time.time()
                #print(f'get_max_adv for {self.num_data} time:' +
                #      f'{time_end-time_start}s!')
                # print(x, x_prime, opt_improve)
                if opt_improve > best_opt_improve:
                    best_opt_improve = opt_improve
                    best_x = x
            if type(best_x) is not list:
                best_x = [best_x]
            print(f'Grid: {best_x} with opt imp {best_opt_improve}.')
            best_x_co_opt = self.query_new_point_co_opt(x_prime)
            co_opt_improve = self.get_max_adv(best_x_co_opt, x_prime)
            print(f'Co_opt: {best_x_co_opt} with opt imp {co_opt_improve}.')
            best_x = best_x_co_opt
        return best_x

    def get_lower_bound(self, x, beta=None):
        # compute the lower bound condition on the current data
        # lists to store diferent converged solutions
        # print(f'Start get_lower_bound with num_data {self.num_data}.')
        start_time = time.time()
        converge_sol_list = []
        converge_val_list = []

        if beta is None:
            beta = self.beta_func(self.num_data)

        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        input_x = [self.x_prime_list[0]] + self.x_list + [x]
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)
        one_arr = np.array(self.one_list)
        input_cov = kernel.K(input_x_arr) + \
            np.eye(self.num_data + 2) * self.numerical_epsilon
        input_cov_inv = np.linalg.inv(input_cov)

        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()
            var_y = opti.variable(
                self.num_data+2)   # variable to be optimized

            if self.num_data > 0:
                min_norm_y = self.get_maxLL_min_norm_inter(x)
                var_y_start = np.append(self.max_LL_y, min_norm_y)
                opti.set_initial(var_y, var_y_start)

            for k in range(self.num_data + 2):
                opti.subject_to(var_y[k] >= self.var_lb)
                opti.subject_to(var_y[k] <= self.var_ub)

            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                           casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            lower_bound = var_y[-1]
            opti.minimize(lower_bound)

            opti.subject_to(
                var_y.T@input_cov_inv@var_y <= self.norm_bound**2
            )

            if self.num_data > 0:
                opti.subject_to(
                    LL >= self.max_LL - beta
                )

            # set solver: 'ipopt'
            opti.solver(
                'ipopt', dict(print_time=False), dict(print_level=False,
                                                      max_iter=100)
            )
            try_solve_limit = 5
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(lower_bound))
                except Exception as e:
                    print(e, ' in optimization_problem.')
                    print(
                        'Lower bound is ' +
                        f'{opti.debug.value(lower_bound)}'
                    )
                    print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(lower_bound))

            best_conv_sol_id = np.argmin(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])

        return best_conv_val

    def get_min_norm_inter(self, x_list, x_prime_list, y_list, x):

        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        if self.num_data == 0:
            return 0

        input_x = [x_prime_list[0]] + x_list
        len_x = len(input_x)
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)
        input_cov = kernel.K(input_x_arr) + \
            np.eye(len_x) * self.numerical_epsilon
        input_cov_inv = np.linalg.inv(input_cov)

        y = np.array(y_list)
        kt = kernel.K(input_x_arr, np.array([x]))

        min_norm_inter_val = kt.T@input_cov_inv@y
        return min_norm_inter_val

    def get_maxLL_min_norm_inter(self, x, is_x_symb=False):

        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        if self.num_data == 0:
            return 0

        if self.num_data > 0:
            old_input_x = [self.x_prime_list[0]] + self.x_list
            old_input_x_arr = np.array(old_input_x)
            if old_input_x_arr.ndim == 1:
                old_input_x_arr = np.expand_dims(old_input_x_arr, axis=1)
            one_arr = np.array(self.one_list)
            old_input_cov = kernel.K(old_input_x_arr) + \
                np.eye(self.num_data + 1) * self.numerical_epsilon
            old_input_cov_inv = np.linalg.inv(old_input_cov)
        else:
            init_y = [0, 0]

        input_x = [self.x_prime_list[0]] + self.x_list
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)
        one_arr = np.array(self.one_list)
        input_cov = kernel.K(input_x_arr) + \
            np.eye(self.num_data + 1) * self.numerical_epsilon
        input_cov_inv = np.linalg.inv(input_cov)

        y = self.max_LL_y
        if is_x_symb:
            kt = self.get_symb_kt(input_x_arr, x)
            min_norm_inter_val = kt.T@(input_cov_inv@y)
        else:
            kt = kernel.K(input_x_arr, np.array([x]))
            min_norm_inter_val = kt.T@input_cov_inv@y
        return min_norm_inter_val

    def get_upper_bound(self, x, beta=None):
        # compute the upper bound condition on the current data
        # lists to store diferent converged solutions
        converge_sol_list = []
        converge_val_list = []

        if beta is None:
            beta = self.beta_func(self.num_data)
        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        if self.num_data > 0:
            old_input_x = [self.x_prime_list[0]] + self.x_list
            old_input_x_arr = np.array(old_input_x)
            if old_input_x_arr.ndim == 1:
                old_input_x_arr = np.expand_dims(old_input_x_arr, axis=1)
            one_arr = np.array(self.one_list)
            old_input_cov = kernel.K(old_input_x_arr) + \
                np.eye(self.num_data +1) * self.numerical_epsilon
            old_input_cov_inv = np.linalg.inv(old_input_cov)
        else:
            init_y = [0, 0]


        input_x = [self.x_prime_list[0]] + self.x_list + [x]
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)
        one_arr = np.array(self.one_list)
        input_cov = kernel.K(input_x_arr) + \
            np.eye(self.num_data +2) * self.numerical_epsilon
        input_cov_inv = np.linalg.inv(input_cov)

        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()
            var_y = opti.variable(self.num_data+2)   # variable to be optimized
            for k in range(self.num_data + 2):
                opti.subject_to(var_y[k] >= self.var_lb)
                opti.subject_to(var_y[k] <= self.var_ub)
            if self.num_data > 0:
                min_norm_y = self.get_maxLL_min_norm_inter(x)
                var_y_start = np.append(self.max_LL_y, min_norm_y)
                opti.set_initial(var_y, var_y_start)

            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                           casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            upper_bound = var_y[-1]
            opti.minimize(-upper_bound)

            opti.subject_to(
                var_y.T@input_cov_inv@var_y <= self.norm_bound**2
            )


            if self.num_data > 0:
                opti.subject_to(
                    LL >= self.max_LL - beta
                )

            # set solver: 'ipopt'
            opti.solver('ipopt', dict(print_time=False), dict(print_level=False,
                                                          max_iter=100))
            try_solve_limit = 5
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(upper_bound))
                except Exception as e:
                    print(e, ' in optimization_problem.')
                    print(
                        'Upper bound is ' +
                        f'{opti.debug.value(upper_bound)}'
                    )
                    print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(upper_bound))

            best_conv_sol_id = np.argmax(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])

        return best_conv_val

    def get_max_adv(self, x, x_prime):
        # compute the optimistic advantage of f(x) - f(x_prime)
        # lists to store diferent converged solutions
        converge_sol_list = []
        converge_val_list = []

        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        if self.num_data > 0:
            old_input_x = self.x_list + self.x_prime_list
            old_input_x_arr = np.array(old_input_x)
            if old_input_x_arr.ndim == 1:
                old_input_x_arr = np.expand_dims(old_input_x_arr, axis=1)
            one_arr = np.array(self.one_list)
            old_input_cov = kernel.K(old_input_x_arr) + \
                np.eye(self.num_data * 2) * self.numerical_epsilon
            old_input_cov_inv = np.linalg.inv(old_input_cov)
        else:
            init_y = [0, 0]

        if self.num_data > 0:
            input_x = [self.x_prime_list[0]] + self.x_list + [x]
        else:
            input_x = [x_prime, x]
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)
        one_arr = np.array(self.one_list)
        input_cov = kernel.K(input_x_arr) + \
            np.eye(self.num_data + 2) * self.numerical_epsilon
        input_cov_inv = np.linalg.inv(input_cov)

        #print(input_cov)
        #print(input_cov_inv)
        #print(input_cov@input_cov_inv)
        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()
            var_y = opti.variable(self.num_data+2)   # variable to be optimized
            if self.num_data > 0:
                min_norm_y = self.get_maxLL_min_norm_inter(x)
                var_y_start = np.append(self.max_LL_y, min_norm_y)
                opti.set_initial(var_y, var_y_start)

            for k in range(self.num_data + 2):
                opti.subject_to(
                    var_y[k] <= self.var_ub
                )
                opti.subject_to(
                    var_y[k] >= self.var_lb
                )
            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                           casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            opti_improve = var_y[-1]-var_y[-2]
            opti.minimize(-opti_improve)

            opti.subject_to(
                var_y.T@input_cov_inv@var_y <= self.norm_bound**2
            )


            if self.num_data > 0:
                opti.subject_to(
                    LL >= self.max_LL - self.beta_func(self.num_data)
                )

            # set solver: 'ipopt'
            opti.solver('ipopt', dict(print_time=False), dict(print_level=False,
                                                          max_iter=100))
            try_solve_limit = 5
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(opti_improve))
                except Exception as e:
                    print(e, ' in optimization_problem.')
                    print(
                        'Optimistic improvement is ' +
                        f'{opti.debug.value(opti_improve)}'
                    )
                    print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(opti_improve))

            best_conv_sol_id = np.argmax(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])

        return best_conv_val

    def update_MLL_with_len_scale(self):
        # compute the maximum likelihood over the historical data
        if self.num_data == 0:
            return None

        # lists to store diferent converged solutions
        converge_sol_list = []
        converge_val_list = []
        converge_len_list = []
        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        input_x = [self.x_prime_list[0]] + self.x_list
        one_arr = np.array(self.one_list)
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)

        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()
            var_y = opti.variable(self.num_data + 1)
            # variable to be optimized
            var_len = opti.variable(self.dim_x)

            input_cov = self.get_symb_K(input_x_arr, var_len) + \
                np.eye(self.num_data + 1) * self.numerical_epsilon

            # input_cov_inv = np.linalg.inv(input_cov)

            old_var_y = self.max_LL_y
            old_num_data = self.num_data - 1
            if self.num_data > 1 and len(old_var_y) == old_num_data +1:
                new_x = self.x_list[-1]
                new_x_prime = self.x_prime_list[-1]
                init_y = self.get_min_norm_inter(
                    self.x_list[:-1],
                    self.x_prime_list[:-1],
                    old_var_y,
                    new_x)
                init_y_prime = self.get_min_norm_inter(
                    self.x_list[:-1],
                    self.x_prime_list[:-1],
                    old_var_y,
                    new_x_prime)

                var_y_start = np.append(
                    old_var_y[:old_num_data+1],
                    init_y
                )

                opti.set_initial(var_y, var_y_start)
                print(self.kernel.lengthscale[0])
                opti.set_initial(var_len, self.kernel.lengthscale)
            for k in range(self.num_data + 1):
                opti.subject_to(
                    var_y[k] <= self.var_ub
                )
                opti.subject_to(
                    var_y[k] >= self.var_lb
                )

            for k in range(self.dim_x):
                print(self.orig_len_scale[k])
                opti.subject_to(
                    var_len[k] <= self.orig_len_scale[k] * 3
                )
                opti.subject_to(
                    var_len[k] >= self.orig_len_scale[k] * 0.2
                )


            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                        casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            opti.minimize(-LL)

            opti.subject_to(
                var_y.T@casadi.solve(input_cov, var_y) <= self.norm_bound**2
            )

            # set solver
            opti.solver('ipopt', dict(print_time=False), dict(print_level=False,
                                                          max_iter=100))
            try_solve_limit = 1
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(LL))
                    converge_len_list.append(sol.value(var_len))
                except Exception as e:
                    print(e, ' in optimization_problem.')
                    print(f'LL is {opti.debug.value(LL)}')
                    print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(LL))
                    converge_len_list.append(opti.debug.value(var_len))

            best_conv_sol_id = np.argmax(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])
            best_conv_sol = converge_sol_list[best_conv_sol_id]

            best_conv_len_scale = converge_len_list[best_conv_sol_id]

            #self.max_LL = best_conv_val
            #self.max_LL_y = best_conv_sol
            #print(f'Old lenthscale: {self.kernel.lengthscale}.')
            print(f'New lengthscale: {best_conv_len_scale}.')
            #for k in range(self.dim_x):
            #    self.kernel.lengthscale[k] = best_conv_len_scale[k]
            #print(self.kernel)
            #old_kernel = self.kernel
            #self.kernel.lengthscale = [
            #    best_conv_len_scale[k] for k in range(self.dim_x)
            #]

            #self.config['kernel'] = self.kernel
            #print(self.kernel.lengthscale)
            #print(self.kernel)
            #print(self.max_LL)
            #print(self.max_LL_y)
            #print(self.config['bounds'])
            #print(self.x_list)
            #print(self.x_prime_list)
            # switch back
            #self.kernel = old_kernel
            #self.config['kernel'] = old_kernel
            #exit()
        return self.max_LL, opti

    def update_MLL(self):
        # compute the maximum likelihood over the historical data
        if self.num_data == 0:
            return None

        if self.num_data % self.update_len_scale_period == 0:
            #pass
            self.update_MLL_with_len_scale()

        # lists to store diferent converged solutions
        converge_sol_list = []
        converge_val_list = []
        converge_norm_square_list = []

        # construct some parameters of the optimization
        config = self.config
        kernel = config['kernel']

        input_x = [self.x_prime_list[0]] + self.x_list
        one_arr = np.array(self.one_list)
        input_x_arr = np.array(input_x)
        if input_x_arr.ndim == 1:
            input_x_arr = np.expand_dims(input_x_arr, axis=1)

        input_cov = kernel.K(input_x_arr) + \
            np.eye(self.num_data + 1) * self.numerical_epsilon

        input_cov_inv = np.linalg.inv(input_cov)

        # construct the optimization problem
        solver = self.solver
        if solver == 'ipopt':
            opti = casadi.Opti()
            var_y = opti.variable(self.num_data + 1)
            # variable to be optimized

            old_var_y = self.max_LL_y
            old_num_data = self.num_data - 1
            if self.num_data > 1 and len(old_var_y) == old_num_data +1:
                new_x = self.x_list[-1]
                new_x_prime = self.x_prime_list[-1]
                init_y = self.get_min_norm_inter(
                    self.x_list[:-1],
                    self.x_prime_list[:-1],
                    old_var_y,
                    new_x)
                init_y_prime = self.get_min_norm_inter(
                    self.x_list[:-1],
                    self.x_prime_list[:-1],
                    old_var_y,
                    new_x_prime)

                var_y_start = np.append(
                    old_var_y[:old_num_data+1],
                    init_y
                )

                opti.set_initial(var_y, var_y_start)

            for k in range(self.num_data + 1):
                opti.subject_to(
                    var_y[k] <= self.var_ub
                )
                opti.subject_to(
                    var_y[k] >= self.var_lb
                )

            LL = sum(var_y[k+1] * one_arr[k] for k in range(self.num_data)) + \
                sum(
                    var_y[k] * (1-one_arr[k])
                    for k in range(self.num_data)) - \
                sum(
                    casadi.log(casadi.exp(var_y[k+1]) +
                        casadi.exp(var_y[k])
                        )
                    for k in range(self.num_data)
                )   # loglikelihood of the historical y

            # set objective
            opti.minimize(-LL)

            norm_square = var_y.T@input_cov_inv@var_y
            opti.subject_to(
                norm_square <= self.norm_bound**2
            )

            # set solver
            opti.solver('ipopt', dict(print_time=False), dict(print_level=False,
                                                          max_iter=100))
            try_solve_limit = 5
            for try_solve_iter in range(try_solve_limit):
                try:
                    sol = opti.solve()
                    converge_sol_list.append(sol.value(var_y))
                    converge_val_list.append(sol.value(LL))
                    converge_norm_square_list.append(sol.value(norm_square))
                except Exception as e:
                    print(e, ' in optimization_problem.')
                    print(f'LL is {opti.debug.value(LL)}')
                    print(f'var_y is {opti.debug.value(var_y)}.')
                    converge_sol_list.append(opti.debug.value(var_y))
                    converge_val_list.append(opti.debug.value(LL))
                    converge_norm_square_list.append(opti.debug.value(norm_square))

            best_conv_sol_id = np.argmax(converge_val_list)
            best_conv_val = np.squeeze(converge_val_list[best_conv_sol_id])
            best_conv_sol = converge_sol_list[best_conv_sol_id]
            best_norm_square = converge_norm_square_list[best_conv_sol_id]
            self.max_LL = best_conv_val
            self.max_LL_y = best_conv_sol
        print(f'Step {self.num_data}: norm {np.sqrt(best_norm_square)}, '+
              f'LL value {self.max_LL},'+
              f'avg {self.max_LL/max(self.num_data, 1)}.')
        return self.max_LL, opti
