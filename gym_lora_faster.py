import bisect
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import pow, ceil
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize


# from matplotlib import pyplot as plt

# class toy_problem:
#     def __init__(self, dim):
#         self.dim = dim
#
#     def fitness(self, x):
#         return [sum(x), 1 - sum(x * x), - sum(x)]
#
#     def gradient(self, x):
#         return pg.estimate_gradient(lambda x: self.fitness(x), x)  # numerical gradient
#
#     def get_nec(self):
#         return 1
#
#     def get_nic(self):
#         return 1
#
#     def get_bounds(self):
#         return ([-1] * self.dim, [1] * self.dim)
#
#     def get_name(self):
#         return "A toy problem"
#
#     def get_extra_info(self):
#         return "\tDimensions: " + str(self.dim)


# class my_constrained_udp():  # baseproblem):
#     def __init__(self, num_nodos, lora):
#         self.N = num_nodos
#         self.num_vars = num_nodos * 49
#         self.lora = lora
#         # super(my_constrained_udp, self).__init__(self.num_vars)
#         # self.set_bounds([0] * self.num_vars, [1] * self.num_vars)
#
#     def get_bounds(self):
#         return [0] * self.num_vars, [1] * self.num_vars
#
#     def get_nic(self):
#         return 0
#
#     def get_nec(self):
#         return self.N
#
#     def fitness(self, x):
#         r_eqs = list()
#         r = lora.get_reward_matricial_alt(x)
#         r_eqs.append(r)
#         for nodo in range(self.N):
#             r_eqs.append(
#                 sum([
#                         x[nodo + self.N + i] for i in range(49)
#                         ]) - 1
#             )
#
#         return r_eqs
#
#     def gradient(self, x):
#         return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


class LoRaWorld:
    def __init__(self, lambdas_, lengths, tx_priorities, power_priorities, snrs, c_opt, alpha):
        """
        :type lambdas_ np.ndarray
        :param lambdas_: generation rate (packets per second)
        :param lengths: lengths, IN BYTES, of the payloads
        :param tx_priorities: priorities of nodes (natural numbers)
        :param power_priorities: priorities of currents drawn (natural numbers)
        :param snrs: SNRs of each node € (-inf, inf)
        :param c_opt: Matrix that indicates the best config for each node (49 x lambdas_.shape[0])
        :param alpha: Parameter to adjust the importance of Throughput vs Current Consumption
        """
        assert lambdas_.shape[0] == lengths.shape[0] == tx_priorities.shape[0] == power_priorities.shape[
            0], 'Incorrect input size'
        self.lambdas_ = np.array(lambdas_)
        self.lengths = np.array(lengths)
        self.tx_priorities = np.array(tx_priorities)
        self.power_priorities = np.array(power_priorities)
        self.alpha = alpha
        self.snrs = np.array(snrs)
        self.N = lambdas_.shape[0]
        self.C = np.zeros((49, self.N))  # :type np.ndarray
        self.C_opt = np.array(c_opt)  # :type np.ndarray
        self.R = 0
        self.BW = 125e3
        self.preamble_symbols = 8
        self.header_length = 13
        self.explicit_header = 1
        # warnings.warn('cambiar aqui')
        self.DC = 0.001  # 0.1%
        # warnings.warn('cambiar aqui')
        self.TDC_MAX = 1000  # 3600 * self.DC
        self.last_packet = None
        self.starting_config = None
        self.ToA = None
        self.factor_l = None

        # for faster gym
        self.future_events = None
        self.time_line = None
        self.t = None
        #

        # for TDC
        self.last_TDC = None
        #

        self.min_lambda = None
        self.max_lambda = None

        self.reporting_consumption = None
        self.min_consumption = None
        self.max_consumption = None

        self.min_tx_priority = None
        self.max_tx_priority = None

        self.min_power_priority = None
        self.max_power_priority = None

        self.alphas = np.array(
            [0, -30.2580, -77.1002, -244.6424, -725.9556, -2109.8064, -4452.3653, -105.1966, -289.8133, -1114.3312,
             -4285.4440, -20771.6945, -98658.1166])
        self.betas = np.array(
            [0, 0.2857, 0.2993, 0.3223, 0.3340, 0.3407, 0.3317, 0.3746, 0.3756, 0.3969, 0.4116, 0.4332, 0.4485])

        self.ToA = np.zeros([49, self.N])
        for c in range(1, 13):
            for nodo in range(self.N):
                self.ToA[[c, c + 12, c + 24, c + 36], nodo] = self.compute_over_the_air_time(self.lengths[nodo],
                                                                                             *self.compute_sf_cr(c))
        nerfs = np.array(
            [-18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -18.1, -11.6, -11.6, -11.6,
             -11.6, -11.6, -11.6, -11.6, -11.6, -11.6, -11.6, -11.6, -11.6, -6., -6., -6., -6., -6., -6., -6., -6., -6.,
             -6., -6., -6., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
        nerfs = np.tile(nerfs, (self.N, 1)).T
        self.snrs_nerfed = (nerfs + self.snrs)
        self.snrs_threshold = self.snrs_nerfed + 6

        self.factor_l = np.zeros([49, self.N])
        for c in range(1, 13):
            for txp_i, txp in enumerate([-4, 2.5, 8.1, 14.1]):
                for nodo in range(self.N):
                    snr = self.snrs[nodo] - (14.1 - txp)
                    prr = self.compute_prr(self.lengths[nodo], *self.compute_sf_cr(c), snr)
                    length = self.lengths[nodo]
                    priority = self.tx_priorities[nodo]
                    lambda_ = self.lambdas_[nodo]

                    self.factor_l[c + (txp_i * 12), nodo] = prr * length * priority * lambda_

        self.tdc_consumption = np.zeros([49, self.N])
        for c in range(1, 13):
            for nodo in range(self.N):
                self.tdc_consumption[[c, c + 12, c + 24, c + 36], nodo] = self.compute_over_the_air_time(52,
                                                                                                         *self.compute_sf_cr(
                                                                                                             c))

        # compute_consumption(self, payload_length, action, pot_tx):
        self.current_consumption = np.zeros([49, self.N])
        for c in range(1, 13):
            for txp_i, txp in enumerate([-4, 2.5, 8.1, 14.1]):
                for nodo in range(self.N):
                    self.current_consumption[c + (txp_i * 12), nodo] = self.compute_consumption(self.lengths[nodo], c,
                                                                                                txp) * \
                                                                       self.power_priorities[nodo]

                    # print('Gym LoRa initialization done')

    # GYM METHODS
    def reset(self, gen_random=False) -> np.ndarray:
        """

        :return: just the observation
        """

        if gen_random:
            starting_config = np.random.randint(1, 13, self.N)
        else:
            starting_config = np.array([self.get_starting_conf(snr, length) for snr, length in
                                        zip(self.snrs + np.random.normal(0, 2, (self.N)), self.lengths)])

        starting_config += (12 * 3)  # max power
        self.reporting_consumption = self.tdc_consumption[starting_config, np.arange(self.N)]

        self.C = np.zeros((49, self.N))
        self.C[starting_config, np.arange(self.N)] = 1
        self.R = self.compute_network_performance()
        self.c_update_state = np.zeros((self.N))

        # NEW!
        self.t = 0
        following_events = -np.log(1.0 - np.random.rand(self.N)) / self.lambdas_
        self.future_events = np.argsort(following_events).tolist()  # type: list
        self.time_line = following_events[self.future_events]  # type: np.ndarray
        self.last_packet = None
        # warnings.warn('cambiar aqui')
        self.last_TDC = self.TDC_MAX  # / 10  # 6 segundos restantes
        #

        self.min_lambda = self.lambdas_.min()
        self.max_lambda = self.lambdas_.max()

        self.min_consumption = self.reporting_consumption.min()
        self.max_consumption = self.reporting_consumption.max()

        self.max_tx_priority = self.tx_priorities.max()
        self.min_tx_priority = self.tx_priorities.min()

        self.max_power_priority = self.power_priorities.max()
        self.min_power_priority = self.power_priorities.min()

        self.b_lambda = None
        self.a_lambda = None
        self.b_consumption = None
        self.a_consumption = None
        self.b_efficiency = None
        self.a_efficiency = None
        self.b_power_priorities = None
        self.a_power_priorities = None
        self.b_tx_priorities = None
        self.a_tx_priorities = None
        # do it for the first time ever
        self.recompute_stats()
        # return self.gather_observation()
        return np.concatenate([np.array([self.last_TDC]), np.zeros((self.N * 2))])

    def step(self, action):
        """
        :param action: {True, False} indicates weather to transmit the update or not
        :return: observation, reward, done, info
        """

        # updating the wall clock, future_events and time_line
        next_t = self.time_line[0]
        elapsed_t = next_t - self.t
        self.t = next_t
        next_event = self.future_events.pop(0)
        self.time_line = self.time_line[1:]

        # generating the new same-type event following poisson distribution
        new_event_t = self.t - np.log(1.0 - np.random.rand()) / self.lambdas_[next_event]
        new_pos = bisect.bisect(self.time_line, new_event_t)

        # inserting in place the new event
        self.time_line = np.insert(self.time_line, new_pos, new_event_t)
        self.future_events.insert(new_pos, next_event)

        assert (self.N - self.c_update_state.sum()) == len(self.future_events)

        penalty = 0
        if action != 0 and self.last_packet is not None:
            if not self.c_update_state[self.last_packet]:
                consumption_TDC = self.reporting_consumption[self.last_packet]

                if self.last_TDC == self.TDC_MAX:
                    # if self.last_TDC >= consumption_TDC:
                    # print('Updating node {}'.format(self.last_packet))
                    self.last_TDC -= consumption_TDC
                    self.C[:, self.last_packet] = self.C_opt[:, self.last_packet]
                    self.R = self.compute_network_performance()
                    self.c_update_state[self.last_packet] = 1

                    # updating normalizing stats
                    self.recompute_stats()

                    # elimino de ambas listas los eventos de este tipo, ya no voy a reportarlos
                    old_pos = self.future_events.index(self.last_packet)
                    self.time_line = np.delete(self.time_line, old_pos)
                    self.future_events.pop(old_pos)

        R = self.R * elapsed_t
        self.last_TDC = min(self.last_TDC + (elapsed_t * self.DC), self.TDC_MAX)

        # para computar la reward en el siguiente step
        self.last_packet = next_event  # self.future_events[0]

        new_obs = np.zeros((self.N))
        new_obs[next_event] = 1
        new_obs = np.concatenate([np.array([self.last_TDC]), self.c_update_state, new_obs])

        done = np.all(self.c_update_state)

        return new_obs, R, done, self.t

    def recompute_stats(self):
        if np.all(self.c_update_state):
            self.b_lambda = None
            self.a_lambda = None
            self.b_consumption = None
            self.a_consumption = None
            self.b_efficiency = None
            self.a_efficiency = None
            self.b_power_priorities = None
            self.a_power_priorities = None
            self.b_tx_priorities = None
            self.a_tx_priorities = None
            return

        selector = np.invert(self.c_update_state.astype(np.bool))

        # FOR LAMBDA
        remaining_lambdas = self.lambdas_[selector]
        max_remaining_lambdas = remaining_lambdas.max()
        min_remaining_lambdas = remaining_lambdas.min()
        if max_remaining_lambdas == min_remaining_lambdas:
            self.b_lambda = 1
        else:
            self.b_lambda = max_remaining_lambdas - min_remaining_lambdas
        self.a_lambda = min_remaining_lambdas / self.b_lambda

        # FOR REPORTING CONSUMPTION
        remaining_consumption = self.reporting_consumption[selector]
        max_remaining_consumption = remaining_consumption.max()
        min_remaining_consumption = remaining_consumption.min()
        if max_remaining_consumption == min_remaining_consumption:
            self.b_consumption = 1
        else:
            self.b_consumption = max_remaining_consumption - min_remaining_consumption

        self.a_consumption = min_remaining_consumption / self.b_consumption

        # FOR TX PRIORITIES
        remaining_tx_priorities = self.tx_priorities[selector]
        max_remaining_tx_priorities = remaining_tx_priorities.max()
        min_remaining_tx_priorities = remaining_tx_priorities.min()
        if max_remaining_tx_priorities == min_remaining_tx_priorities:
            self.b_tx_priorities = 1
        else:
            self.b_tx_priorities = max_remaining_tx_priorities - min_remaining_tx_priorities
        self.a_tx_priorities = min_remaining_tx_priorities / self.b_tx_priorities

        # FOR POWER PRIORITIES
        remaining_power_priorities = self.power_priorities[selector]
        max_remaining_power_priorities = remaining_power_priorities.max()
        min_remaining_power_priorities = remaining_power_priorities.min()
        if max_remaining_power_priorities == min_remaining_power_priorities:
            self.b_power_priorities = 1
        else:
            self.b_power_priorities = max_remaining_power_priorities - min_remaining_power_priorities
        self.a_power_priorities = min_remaining_power_priorities / self.b_power_priorities

        # FOR EFFICIENCY

        C_backup = np.array(self.C)
        current_R = self.R

        self.efficiencies = np.full(self.N, np.nan)

        for node in np.where(selector)[0]:
            self.C = np.array(C_backup)
            self.C[:, node] = self.C_opt[:, node]
            R = self.compute_network_performance()
            self.efficiencies[node] = (R - current_R) / 10000

        self.C = np.array(C_backup)

        max_remaining_rs = np.nanmax(self.efficiencies)
        min_remaining_rs = np.nanmin(self.efficiencies)

        if max_remaining_rs == min_remaining_rs:
            self.b_efficiency = 1
        else:
            self.b_efficiency = max_remaining_rs - min_remaining_rs

        self.a_efficiency = min_remaining_rs / self.b_efficiency

    # HELPERS
    def gather_last_rewards(self, til=864000):
        """
        Gather last rewards until 1 hour
        :param til: The time horizon - default: 10 days
        :return: the gathered reward until the time horizon
        """

        current_reward = self.compute_network_performance()
        remaining_time = (til - min(self.t, til))
        return current_reward * remaining_time

    # poner los breakpoints dentro de funciones (es decir: no en el main)
    def get_reward_matricial_alt(self, X):
        C = X.reshape(49, self.N)
        # lambdas_agg = self.lambdas_.dot(C.T)
        # lambdas_agg_rel = np.tile(lambdas_agg, (self.N, 1)).T - (C * self.lambdas_)
        # lambdas_agg_rel_folded = (lambdas_agg_rel[1:7, :] + lambdas_agg_rel[7:13, :] + lambdas_agg_rel[13:19, :]
        #                           + lambdas_agg_rel[19:25, :] + lambdas_agg_rel[25:31, :] + lambdas_agg_rel[31:37, :]
        #                           + lambdas_agg_rel[37:43, :] + lambdas_agg_rel[43:49, :])
        # lambdas_any = np.vstack([np.zeros((1, self.N)), np.tile(lambdas_agg_rel_folded, (8, 1))])
        # # lambdas_agg_rel_folded_stacked <- cualquier nodo ha transmitido en el mismo SF

        lambdas_any = np.zeros((49, self.N))
        for c in range(1, 49):
            for n in range(self.N):
                n_c = (c - 1) % 6  # SF
                config_filter = np.zeros((48, self.N))
                config_filter[[n_c, n_c + 6, n_c + 12, n_c + 18, n_c + 24, n_c + 30, n_c + 36, n_c + 42], :] = 1
                config_filter[:, n] = 0
                lambdas_any[c, n] = (config_filter * C[1:, :] * self.lambdas_ * self.ToA[1:, :]).sum()

        lambdas_higher = np.zeros((49, self.N))
        for c in range(1, 49):
            for n in range(self.N):
                n_c = (c - 1) % 6  # SF
                matrix_filter = (self.snrs_threshold >= self.snrs_nerfed[c - 1, n]).astype(np.int)
                matrix_filter[:, n] = 0
                config_filter = np.zeros((48, self.N))
                config_filter[[n_c, n_c + 6, n_c + 12, n_c + 18, n_c + 24, n_c + 30, n_c + 36, n_c + 42], :] = 1
                lambdas_higher[c, n] = (matrix_filter * config_filter * C[1:, :] * self.lambdas_).sum()

        lambdas_higher *= self.ToA

        lambdas_tot = lambdas_any + lambdas_higher

        R_T = (self.factor_l * C * np.exp(-lambdas_tot)).sum()
        R_C = (self.current_consumption * C * self.lambdas_).sum()
        R = self.alpha * R_T - (1 - self.alpha) * R_C

        return -R / self.N

    def get_positive_reward_matricial(self, X):
        return -self.get_reward_matricial(X)

    def get_reward_matricial_13_ES(self, X):
        if np.any(np.isnan(X)):
            print('Found NaN')
            X[:, :] = 1 / 13

        C = X.reshape(13, self.N)

        assert np.allclose(C.sum(axis=0), 1)
        assert np.all(C.max(axis=0) <= 1)
        assert np.all(C.min(axis=0) >= 0)

        Y = np.zeros((49, self.N))
        Y[0, :] = C[0, :]
        Y[1 + 12 * 3: 1 + 12 * 4] = C[1:13, :]
        return - self.get_reward_matricial(Y)

    def get_reward_matricial_13(self, X):
        C = X.reshape(13, self.N)
        Y = np.zeros((49, self.N))
        Y[0, :] = C[0, :]
        Y[1 + 12 * 3: 1 + 12 * 4] = C[1:13, :]
        return self.get_reward_matricial(Y)

    def get_reward_matricial(self, X):
        # global N, lengths, lambdas_, priorities, SNRs, ToA, factor_l
        C = X.reshape(49, self.N)

        lambdas_agg = self.lambdas_.dot(C.T)  # self.lambdas_[:self.N].dot(C.T)

        lambdas_agg_folded = np.hstack([lambdas_agg[1:7] + lambdas_agg[7:13] + lambdas_agg[13:19] + lambdas_agg[
                                                                                                    19:25] + lambdas_agg[
                                                                                                             25:31] + lambdas_agg[
                                                                                                                      31:37] + lambdas_agg[
                                                                                                                               37:43] + lambdas_agg[
                                                                                                                                        43:49],
                                        lambdas_agg[1:7] + lambdas_agg[7:13] + lambdas_agg[13:19] + lambdas_agg[
                                                                                                    19:25] + lambdas_agg[
                                                                                                             25:31] + lambdas_agg[
                                                                                                                      31:37] + lambdas_agg[
                                                                                                                               37:43] + lambdas_agg[
                                                                                                                                        43:49]])
        # lambdas_agg_folded = np.tile(lambdas_agg[1:].reshape(8, 6).sum(axis=0), 2)
        exp_factor = -2 * self.ToA * np.vstack([np.zeros((1, self.N)), np.tile(lambdas_agg_folded, (self.N, 4)).T])

        R_T = (self.factor_l * C * np.exp(exp_factor)).sum()
        if self.alpha == 1:
            R_C = 0
        else:
            R_C = (self.current_consumption * C * self.lambdas_).sum()

        R = self.alpha * R_T - (1 - self.alpha) * R_C

        return -R / self.N

    def find_optimal_by_adr(self):
        """
    If NStep > 0 the data rate can be increased and/or power reduced. If Nstep < 0, power can be increased (to the max.).

    For NStep > 0, first the data rate is increased (by Nstep) until DR5 is reached. If the number of steps < Nstep, the remainder is used to decrease the TXpower by 3dBm per step, until TXmin is reached. TXmin = 2 dBm for EU868.
        """

        # SNRmargin = SNRm – RequiredSNR(DR) - margin_db

        # simulate 20 readings in an std=4 LNSPL model
        SNRs_max = self.snrs + np.random.normal(0, 4, (20, self.N)).max(axis=0)
        for nodo in range(self.N):
            config = self.get_conf_by_adr(SNRs_max[nodo])
            self.C_opt[:, nodo] = np.zeros(49)
            self.C_opt[config, nodo] = 1

        return self.C_opt, None

    def find_genetic_optimal(self):
        from es import PEPG
        NPOPULATION = 101
        NPARAMS = 49
        WORKERS = 1

        initial_model = np.random.rand(49, self.N)
        initial_model /= initial_model.sum(axis=0)

        # solver = CMAES(NPARAMS * self.N,
        #               popsize=NPOPULATION,
        #               sigma_init=1,
        #               x0=initial_model.flatten(),
        #               weight_decay=0
        #               )

        solver = PEPG(NPARAMS * self.N,  # number of model parameters
                      sigma_init=1,  # initial standard deviation
                      learning_rate=1e-3,  # learning rate for standard deviation
                      elite_ratio=0.25,
                      popsize=NPOPULATION,  # population size
                      average_baseline=False,  # set baseline to average of batch
                      weight_decay=0.00,  # weight decay coefficient
                      rank_fitness=True,  # use rank rather than fitness numbers
                      forget_best=False)  # don't keep the historical best solution)

        for j in range(100):
            solutions = solver.ask()
            solutions.shape = (NPOPULATION, 49, self.N)
            solutions = [(solutions[_] - solutions[_].min(axis=0)) / solutions.max(axis=0) for _ in range(NPOPULATION)]
            solutions = [solutions[_] / solutions[_].sum(axis=0) for _ in range(NPOPULATION)]

            # no se está haciendo bien esto
            if WORKERS > 1:
                with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                    p = executor.map(self.get_reward_matricial, solutions)
                    fitness_list = np.array(list(p))
            else:
                fitness_list = list()
                for i in range(len(solutions)):
                    fitness_list.append(self.get_reward_matricial(solutions[i]))

            solver.solutions = [s.flatten() for s in solutions]
            solver.tell(fitness_list)
            # solver.tell(fitness_list, [s.flatten() for s in solutions])
            result = solver.result()
            print("max fitness at iteration", (j + 1), result[1])
            print("Average fitness of top 25% at iteration", (j + 1),
                  np.sort(fitness_list)[-int(NPOPULATION / 4):].mean())
            print('ok')

        print('end')
        exit(-1)

    def find_optimal_c(self, passes=1, verbose=True):
        assert 0 < passes < 100

        constraints = list()
        for nodo in range(self.N):
            constraints.append(
                {
                    'type': 'eq',
                    'fun': lambda x, nodo=nodo: x[nodo + self.N * 0] + x[nodo + self.N * 1] + x[nodo + self.N * 2] + x[
                        nodo + self.N * 3] + x[nodo + self.N * 4] + x[nodo + self.N * 5] + x[nodo + self.N * 6] + x[
                                                    nodo + self.N * 7] + x[nodo + self.N * 8] + x[nodo + self.N * 9] +
                                                x[nodo + self.N * 10] + x[nodo + self.N * 11] + x[nodo + self.N * 12] +
                                                x[nodo + self.N * 13] + x[nodo + self.N * 14] + x[nodo + self.N * 15] +
                                                x[nodo + self.N * 16] + x[nodo + self.N * 17] + x[nodo + self.N * 18] +
                                                x[nodo + self.N * 19] + x[nodo + self.N * 20] + x[nodo + self.N * 21] +
                                                x[nodo + self.N * 22] + x[nodo + self.N * 23] + x[nodo + self.N * 24] +
                                                x[nodo + self.N * 25] + x[nodo + self.N * 26] + x[nodo + self.N * 27] +
                                                x[nodo + self.N * 28] + x[nodo + self.N * 29] + x[nodo + self.N * 30] +
                                                x[nodo + self.N * 31] + x[nodo + self.N * 32] + x[nodo + self.N * 33] +
                                                x[nodo + self.N * 34] + x[nodo + self.N * 35] + x[nodo + self.N * 36] +
                                                x[nodo + self.N * 37] + x[nodo + self.N * 38] + x[nodo + self.N * 39] +
                                                x[nodo + self.N * 40] + x[nodo + self.N * 41] + x[nodo + self.N * 42] +
                                                x[nodo + self.N * 43] + x[nodo + self.N * 44] + x[nodo + self.N * 45] +
                                                x[nodo + self.N * 46] + x[nodo + self.N * 47] + x[
                                                    nodo + self.N * 48] - 1
                }
            )

        bounds = list()
        for x in range(self.N * 49):
            bounds.append((0, 1))

        # options = {'disp': False, 'maxiter': 1e6, 'ftol': 1e-3, 'iprint': 0}
        if verbose:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-4, 'iprint': 2}
        else:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-4, 'iprint': 0}

        max_R = -float('inf')
        for _ in range(int(passes)):
            initial_value = np.random.rand(49 * self.N)
            initial_value.shape = (49, self.N)
            for nodo in range(self.N):
                initial_value[:, nodo] /= initial_value[:, nodo].sum()

            res = optimize.minimize(self.get_reward_matricial, initial_value, method='SLSQP',
                                    bounds=bounds, constraints=constraints, options=options)

            if (-res.fun) > max_R:
                max_R = -res.fun
                self.C_opt = res.x.reshape(49, self.N)
                print('New max found')

        assert max_R > 0, 'Debe ser mayor que 0'
        return self.C_opt, max_R

    def map_solutions(self, solutions, population):
        solutions.shape = (population, 13, self.N)
        solutions = [(solutions[_] - solutions[_].min(axis=0)) / solutions[_].max(axis=0) for _ in range(population)]
        solutions = [solutions[_] / solutions[_].sum(axis=0) for _ in range(population)]
        return solutions

    def find_optimal_c_13_ES(self, solver, population=101, iters=100):
        WORKERS = 1
        best_solution = None

        for j in range(iters):
            solutions = solver.ask()

            solutions = self.map_solutions(solutions, population)

            if WORKERS > 1:
                with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                    p = executor.map(self.get_reward_matricial_13_ES, solutions)
                    fitness_list = np.array(list(p))
            else:
                fitness_list = list()
                for i in range(len(solutions)):
                    fitness_list.append(self.get_reward_matricial_13_ES(solutions[i]))

            solver.tell(fitness_list, solutions)

            best_solution_idx = np.argmax(fitness_list)
            best_solution = solutions[best_solution_idx]

            if (j % 100) == 0:
                print("max fitness at iteration", (j + 1), fitness_list[best_solution_idx])

        Y = np.zeros((49, self.N))
        Y[0, :] = best_solution[0, :]
        Y[1 + 12 * 3: 1 + 12 * 4] = best_solution[1:13, :]
        self.C_opt = Y
        return Y

    def find_optimal_c_13(self, passes=1, verbose=True, initial=None):
        assert 0 < passes < 100

        constraints = list()
        for nodo in range(self.N):
            constraints.append(
                {
                    'type': 'eq',
                    'fun': lambda x, nodo=nodo: x[nodo + self.N * 0] + x[nodo + self.N * 1] + x[nodo + self.N * 2] + x[
                        nodo + self.N * 3] + x[nodo + self.N * 4] + x[nodo + self.N * 5] + x[nodo + self.N * 6] + x[
                                                    nodo + self.N * 7] + x[nodo + self.N * 8] + x[nodo + self.N * 9] +
                                                x[nodo + self.N * 10] + x[nodo + self.N * 11] + x[
                                                    nodo + self.N * 12] - 1
                }
            )

        bounds = list()
        for x in range(self.N * 13):
            bounds.append((0, 1))

        # options = {'disp': False, 'maxiter': 1e6, 'ftol': 1e-3, 'iprint': 0}
        if verbose:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-5, 'iprint': 2}
        else:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-5, 'iprint': 0}

        max_R = -float('inf')
        for _ in range(int(passes)):
            # initial_value = self.find_optimal_by_adr()[0][1 + 12 * 3: 1 + 12 * 4, :]
            if initial is None:
                initial_value = np.random.rand(13 * self.N)
                initial_value.shape = (13, self.N)
            else:
                if initial.shape[0] > 13:
                    initial_value = np.zeros((13, self.N))
                    initial_value[0, :] = initial[0, :]
                    initial_value[1:13, :] = initial[1 + 12 * 3:1 + 12 * 4]
                else:
                    initial_value = initial

            for nodo in range(self.N):
                initial_value[:, nodo] /= initial_value[:, nodo].sum()

            res = optimize.minimize(self.get_reward_matricial_13, initial_value, method='SLSQP',
                                    bounds=bounds, constraints=constraints, options=options)

            if (-res.fun) > max_R:
                max_R = -res.fun
                # aqui meter ya los ceros

                C = res.x.reshape(13, self.N)
                Y = np.zeros((49, self.N))
                Y[0, :] = C[0, :]
                Y[1 + 12 * 3: 1 + 12 * 4, :] = C[1:13, :]

                self.C_opt = Y
                print('New max found')

        assert max_R > 0, 'Debe ser mayor que 0'
        return self.C_opt, max_R

    @staticmethod
    def compute_current_drawn(over_the_air_time, pot_tx):
        consumptions = {
            -4: 17.3,
            2.5: 24.7,
            8.1: 31.2,
            14.1: 39.9,
        }

        assert pot_tx in consumptions
        draw_ma = consumptions[pot_tx]
        return over_the_air_time * draw_ma

    def compute_consumption(self, payload_length, action, pot_tx):
        """
        Computes current consumption derived from transmitting the packet (in mA)
        :param payload_length: Length of the transmitted packet in bytes (only the payload)
        :param action: Action carried out, from 0 to 12
        :param pot_tx: Transmitting power € {-4, 2.5, 8.1, 14.1}
        :return:
        """
        sf, cr = self.compute_sf_cr(action)
        ToA = self.compute_over_the_air_time(payload_length, sf, cr)
        return self.compute_current_drawn(ToA, pot_tx)

    @staticmethod
    def compute_sf_cr(action):
        return np.array([0, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12])[action], \
               np.array([0, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7])[action]

    def compute_prr(self, packet_length, sf, cr, snr):
        # packet_length in bytes
        if sf == 0 and cr == 0:
            return 0

        assert cr in (5, 7), "CR not implemented yet"
        assert 7 <= sf <= 12, "Invalid SF"

        action = (sf - 6) + [0, 6][cr == 7]
        alfa = self.alphas[action]
        beta = self.betas[action]
        ber = np.power(10, alfa * np.exp(beta * snr))
        return np.power(1 - ber, packet_length * 8)

    def get_starting_conf(self, snr, packet_length):
        ordered_configs = [0, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12]

        conf = 12
        for conf_i in range(1, 13):
            conf = ordered_configs[conf_i]
            sf, cr = self.compute_sf_cr(conf)
            per = self.compute_prr(packet_length, sf, cr, snr)
            if per > 0.99:
                return conf

        return conf

    @staticmethod
    def get_conf_by_adr(snr):
        # returns the config as an index from 1 to 12

        required_SNR = {7: 2.5,  # this are the required SNR values for each SF plus a 10dB margin
                        8: 0,  # see https://github.com/TheThingsNetwork/ttn/issues/265
                        9: -2.5,
                        10: -5,
                        11: -7.5,
                        12: -10
                        }

        best = 12
        for sf, sf_margin in required_SNR.items():
            if sf_margin < snr:
                best = sf
                break

        chosen_power_index = 3
        power_margins = (6, 5.6, 6.5)  # the transmission Power we lose in each TXPOWER step we decrease
        if best == 7:
            gross_margin = snr - 2.5
            for pm in power_margins:
                if gross_margin > pm:
                    gross_margin -= pm
                    chosen_power_index -= 1
                else:
                    break

        CR = 0  # 0 -> CR=4/5 and 1 -> CR=4/7

        best -= 6
        best += (6 * CR)
        best += (12 * chosen_power_index)  # max power

        return best

    def compute_over_the_air_time(self, payload_length, sf, cr, continuous=False):
        # payload_length in bytes

        if sf == 0 and cr == 0:
            return 0

        assert 7 <= sf <= 12
        assert 5 <= cr <= 7
        de = 1 if sf >= 11 else 0
        # http://forum.thethingsnetwork.org/t/spreadsheet-for-lora-airtime-calculation/1190/15
        t_sym = pow(2, sf) / self.BW * 1000  # symbol time in ms
        t_preamble = (self.preamble_symbols + 4.25) * t_sym  # over the air time of the preamble
        if continuous:
            payload_symbol_number = 8 + (((8 * (payload_length + self.header_length) - 4 * sf + 28 + 16 - 20 * (
                1 - self.explicit_header)) / (4 * (sf - 2 * de))) * cr)
        else:
            payload_symbol_number = 8 + max([(ceil(
                (8 * (payload_length + self.header_length) - 4 * sf + 28 + 16 - 20 * (1 - self.explicit_header)) / (
                    4 * (sf - 2 * de))) * cr), 0])  # number of symbols of the payload

        t_payload = payload_symbol_number * t_sym  # payload time in ms
        t_packet = t_preamble + t_payload

        return t_packet / 1000  # expressed in seconds

    def get_transmittable(self, node):
        return 1
        # d_t = 1e-3
        # lambda_ = self.lambdas_[node]
        # length = self.lengths[node]
        # config = self.C[:, node]
        # transmittables_factors = np.array(
        #     [((self.compute_over_the_air_time(length, *self.compute_sf_cr(action)) / self.DC / d_t) - 1) * (
        #         1 - exp(-lambda_ * config[action] * d_t)) for action in np.arange(13)])
        # transmittables_factors[0] = 1
        #
        # return 1 / transmittables_factors.sum()

    def get_effective_lambda(self, node):
        lambda_ = self.lambdas_[node]
        return lambda_ * self.get_transmittable(node)

    def compute_network_performance(self):
        return -self.get_reward_matricial(self.C)

    def compute_network_performance_precise(self):
        return -self.get_reward_matricial_alt(self.C)

    def fixed_length_input(self, observation, extended=False):
        l_input = observation.shape[0]
        l_output = 9  # lambda, lambda_rel, DR, DR_rel, C, C_rel, TDC, TDC_%, Priority_TX_Rel, Priority_Power_rel
        obs_half = int((l_input - 1) / 2)
        assert (l_input - 1) % 2 == 0, "Should be 2XN + 1"
        TDC = observation[0] / self.TDC_MAX
        non_updated = np.where(observation[1:obs_half + 1] == 0)[0]
        generated_packet = np.where(observation[obs_half + 1:] == 1)[0]
        if len(generated_packet) == 0:  # a packet has NOT been generatesd
            if extended:
                return np.zeros(l_output)
            else:
                return np.array([0, 0, TDC])

        generated_packet = generated_packet[0]
        if generated_packet not in non_updated:  # a node that has already been updated wants to get updated AGAIN :[
            if extended:
                return np.zeros(l_output)
            else:
                return np.array([0, 0, TDC])
        elif non_updated.shape[0] == 1:
            if extended:
                return np.ones(l_output)
            else:
                return np.array([1, 1, TDC])

        pos_lambda = self.lambdas_[generated_packet] / self.b_lambda - self.a_lambda
        pos_consumption = self.reporting_consumption[generated_packet] / self.b_consumption - self.a_consumption
        pos_tx_priority = self.tx_priorities[generated_packet] / self.b_tx_priorities - self.a_tx_priorities
        pos_power_priority = self.power_priorities[generated_packet] / self.b_power_priorities - self.a_power_priorities
        pos_efficiency = self.efficiencies[generated_packet] / self.b_efficiency - self.a_efficiency

        # remaining_lambdas = self.lambdas_[non_updated]
        # max_remaining_lambdas = remaining_lambdas.max()
        # min_remaining_lambdas = remaining_lambdas.min()
        #
        # pos_lambda_bis = (self.lambdas_[generated_packet] - min_remaining_lambdas) / \
        #              (max_remaining_lambdas - min_remaining_lambdas)
        #
        # np.testing.assert_almost_equal(pos_lambda, pos_lambda_bis, 5)
        #
        # remaining_consumption = self.reporting_consumption[non_updated]
        # max_remaining_consumption = remaining_consumption.max()
        # min_remaining_consumption = remaining_consumption.min()
        # if max_remaining_consumption == min_remaining_consumption:
        #     diff_remaining_consumption = 1
        # else:
        #     diff_remaining_consumption = (max_remaining_consumption - min_remaining_consumption)
        #
        # pos_consumption_bis = (self.reporting_consumption[generated_packet] - min_remaining_consumption) / \
        #                   diff_remaining_consumption
        #
        # np.testing.assert_almost_equal(pos_consumption, pos_consumption_bis, 5)
        #
        # remaining_tx_priorities = self.tx_priorities[non_updated]
        # max_remaining_tx_priorities = remaining_tx_priorities.max()
        # min_remaining_tx_priorities = remaining_tx_priorities.min()
        # pos_tx_priority_bis = (self.tx_priorities[generated_packet] - min_remaining_tx_priorities) / \
        #                   (max_remaining_tx_priorities - min_remaining_tx_priorities)
        #
        # np.testing.assert_almost_equal(pos_tx_priority, pos_tx_priority_bis, 5)
        #
        # remaining_power_priorities = self.power_priorities[non_updated]
        # max_remaining_power_priorities = remaining_power_priorities.max()
        # min_remaining_power_priorities = remaining_power_priorities.min()
        # pos_power_priority_bis = (self.power_priorities[generated_packet] - min_remaining_power_priorities) / \
        #                      (max_remaining_power_priorities - min_remaining_power_priorities)
        #
        # np.testing.assert_almost_equal(pos_power_priority, pos_power_priority_bis, 5)
        #
        # C_backup = np.array(self.C)
        # Rs = list()
        # current_R = self.compute_network_performance()
        # for node in non_updated:
        #     self.C = np.array(C_backup)
        #     self.C[:, node] = self.C_opt[:, node]
        #     R = self.compute_network_performance()
        #     Rs.append(R - current_R)
        # self.C = np.array(C_backup)
        # Rs = np.array(Rs)
        # Rs /= 10000
        #
        # pos_in_non_updated = np.where(non_updated == generated_packet)[0][0]
        # max_remaining_rs = Rs.max()
        # min_remaining_rs = Rs.min()
        #
        # if max_remaining_rs == min_remaining_rs:
        #     diff_remaining_efficiency = 1
        # else:
        #     diff_remaining_efficiency = (max_remaining_rs - min_remaining_rs)
        #
        # pos_efficiency_bis = (Rs[pos_in_non_updated] - min_remaining_rs) / diff_remaining_efficiency
        #
        # np.testing.assert_almost_equal(pos_efficiency, pos_efficiency_bis, 5)

        #
        # if extended:
        #     return np.array([self.lambdas_[generated_packet], pos_lambda,
        #                      Rs[pos_in_non_updated], pos_efficiency,
        #                      self.reporting_consumption[generated_packet], pos_consumption,
        #                      TDC,
        #                      pos_tx_priority,
        #                      pos_power_priority
        #                      ])

        if extended:
            return np.array([self.lambdas_[generated_packet], pos_lambda,
                             self.efficiencies[generated_packet], pos_efficiency,
                             self.reporting_consumption[generated_packet], pos_consumption,
                             TDC,
                             pos_tx_priority,
                             pos_power_priority
                             ])
        else:
            return np.array([1 - pos_lambda, pos_efficiency, TDC])



def represent_sfs(global_config, save_path=None, show=False, title=""):
    assert global_config.shape[0] == 12
    utilization = (global_config[:6, :] + global_config[6:, :]).sum(axis=1)
    norm_coeff = utilization.sum()
    utilization /= norm_coeff
    fig, ax = plt.subplots()
    ind = range(6)
    bar = ax.bar(ind, utilization, color='white', edgecolor='black', hatch="\\")
    # for b in bar:
    #     b.set_hatch('\\')
    ax.set_xticklabels(['SF{}'.format(i) for i in range(6, 13)])
    ax.set_xlabel('Spreading Factors', fontsize=16)
    ax.set_ylabel('Percentage utilization', fontsize=16)
    ax.set_title(title, fontsize=18, y=1.05)
    ax.set_ylim([0, 1])
    plt.tight_layout()

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        tick.label.set_rotation(0)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        tick.label.set_rotation(0)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print("ok")
    # lora.C[1 + 12 * 3: 1 + 12 * 4, :]

def experiment_add_remove(num_nodos=10, seed=0):
    np.random.seed(seed)

    min_rate = 30
    max_rate = 30.0001  # vamos a modelar el T medio entre paquetes, y la lambda como inversa de esto
    lambdas_ = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, num_nodos) + min_rate)
    lengths = np.random.randint(29, 30, (lambdas_.shape[0]))

    min_priority = 0.999
    max_priority = 1
    priorities = (max_priority - min_priority) * np.random.random(num_nodos) + min_priority

    min_current_priority = 0.999
    max_current_priority = 1
    current_priorities = (max_current_priority - min_current_priority) * np.random.random(
        num_nodos) + min_current_priority

    SNRs = np.random.randint(-23, -5, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    C_opt = np.zeros((49, lambdas_.shape[0]))
    lora = LoRaWorld(lambdas_, lengths, priorities, current_priorities, SNRs, C_opt, 1)

    t1 = time.time()
    lora.find_optimal_c_13()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora.C = lora.C_opt
    print(lora.compute_network_performance())
    represent_sfs(lora.C[1 + 12 * 3: 1 + 12 * 4, :], save_path='/home/ruben/Dropbox/Tesis Rubén/LoRaCoordination/Article/Images/LowRes/lora_sf_distribution_1.png', title="Spreading Factor utilization for N={} nodes".format(num_nodos))


    # adding more nodes
    new_added_nodes = int(round(num_nodos * 0.2))
    print("Adding {} new nodes".format(new_added_nodes))
    new_lambdas = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, new_added_nodes) + min_rate)
    new_lengths = np.random.randint(15, 30, (new_lambdas.shape[0]))
    new_priorities = (max_priority - min_priority) * np.random.random(new_added_nodes) + min_priority
    new_current_priorities = (max_current_priority - min_current_priority) * np.random.random(
        new_added_nodes) + min_current_priority
    new_SNRs = np.random.randint(-23, 0, (new_added_nodes,))

    lambdas_2 = np.append(lambdas_, new_lambdas)
    lengths2 = np.append(lengths, new_lengths)
    priorities2 = np.append(priorities, new_priorities)
    current_priorities2 = np.append(current_priorities, new_current_priorities)
    SNRs2 = np.append(SNRs, new_SNRs)
    C_opt2 = np.zeros((49, lambdas_.shape[0]))
    lora2 = LoRaWorld(lambdas_2, lengths2, priorities2, current_priorities2, SNRs2, C_opt2, 1)

    t1 = time.time()
    lora2.find_optimal_c_13()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora2.C = lora2.C_opt
    print(lora2.compute_network_performance())
    represent_sfs(lora2.C[1 + 12 * 3: 1 + 12 * 4, :], save_path='/home/ruben/Dropbox/Tesis Rubén/LoRaCoordination/Article/Images/LowRes/lora_sf_distribution_2.png', title="Spreading Factor utilization for N={} nodes".format(lambdas_2.shape[0]))

    # adding even more nodes
    new_added_nodes = int(round(num_nodos * 0.2))
    print("Adding {} extra nodes".format(new_added_nodes))
    new_lambdas = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, new_added_nodes) + min_rate)
    new_lengths = np.random.randint(15, 30, (new_lambdas.shape[0]))
    new_priorities = (max_priority - min_priority) * np.random.random(new_added_nodes) + min_priority
    new_current_priorities = (max_current_priority - min_current_priority) * np.random.random(
        new_added_nodes) + min_current_priority
    new_SNRs = np.random.randint(-23, 0, (new_added_nodes,))

    lambdas_3 = np.append(lambdas_2, new_lambdas)
    lengths3 = np.append(lengths2, new_lengths)
    priorities3 = np.append(priorities2, new_priorities)
    current_priorities3 = np.append(current_priorities2, new_current_priorities)
    SNRs3 = np.append(SNRs2, new_SNRs)
    C_opt3 = np.zeros((49, lambdas_.shape[0]))
    lora3 = LoRaWorld(lambdas_3, lengths3, priorities3, current_priorities3, SNRs3, C_opt3, 1)

    t1 = time.time()
    lora3.find_optimal_c_13()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora3.C = lora3.C_opt
    print(lora3.compute_network_performance())
    represent_sfs(lora3.C[1 + 12 * 3: 1 + 12 * 4, :], save_path='/home/ruben/Dropbox/Tesis Rubén/LoRaCoordination/Article/Images/LowRes/lora_sf_distribution_3.png', title="Spreading Factor utilization for N={} nodes".format(lambdas_3.shape[0]))


    # adding even MUCH more nodes
    new_added_nodes = int(round(num_nodos * 0.2))
    print("Adding {} extra nodes".format(new_added_nodes))
    new_lambdas = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, new_added_nodes) + min_rate)
    new_lengths = np.random.randint(15, 30, (new_lambdas.shape[0]))
    new_priorities = (max_priority - min_priority) * np.random.random(new_added_nodes) + min_priority
    new_current_priorities = (max_current_priority - min_current_priority) * np.random.random(
        new_added_nodes) + min_current_priority
    new_SNRs = np.random.randint(-23, 0, (new_added_nodes,))

    lambdas_4 = np.append(lambdas_3, new_lambdas)
    lengths4 = np.append(lengths3, new_lengths)
    priorities4 = np.append(priorities3, new_priorities)
    current_priorities4 = np.append(current_priorities3, new_current_priorities)
    SNRs4 = np.append(SNRs3, new_SNRs)
    C_opt4 = np.zeros((49, lambdas_.shape[0]))
    lora4 = LoRaWorld(lambdas_4, lengths4, priorities4, current_priorities4, SNRs4, C_opt4, 1)

    t1 = time.time()
    lora4.find_optimal_c_13()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora4.C = lora4.C_opt
    print(lora4.compute_network_performance())
    represent_sfs(lora4.C[1 + 12 * 3: 1 + 12 * 4, :], save_path='/home/ruben/Dropbox/Tesis Rubén/LoRaCoordination/Article/Images/LowRes/lora_sf_distribution_4.png', title="Spreading Factor utilization for N={} nodes".format(lambdas_4.shape[0]))

    exit(-1)


def represent_sfs_crs(global_config, save_path=None, show=True):
    assert global_config.shape[0] == 12
    utilization = global_config.sum(axis=1)
    norm_coeff = utilization.sum()
    utilization /= norm_coeff

    # fig, ax = plt.subplots()
    ind = np.arange(12)
    plt.bar(ind, utilization)

    # ax.set_xticklabels(['SF{} CR=4/5'.format(i) for i in range(6, 13)] + ['SF {} CR=4/7'.format(i) for i in range(6, 13)])
    plt.xlabel('SF and CR')
    plt.xticks(ind -0.5, ['SF{} CR=4/5'.format(i) for i in range(7, 13)] + ['SF{} CR=4/7'.format(i) for i in range(7, 13)], rotation=45)
    plt.ylabel('Percentage utilization')

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    print("ok")


def experiment_vary_importance(num_nodos=10, seed=0):
    np.random.seed(seed)

    lambdas_ = np.full((num_nodos), 1 / 60)
    lengths = np.full((num_nodos), 30)
    priorities = np.full((num_nodos), 1)
    current_priorities = np.full((num_nodos), 1)

    SNRs = np.full((num_nodos), -10) # np.random.randint(-23, 0, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    for p in [1, 30]: # np.linspace(1, 100, 10, endpoint=True):
        C_opt = np.zeros((49, lambdas_.shape[0]))
        priorities[0] = p
        lora = LoRaWorld(lambdas_, lengths, priorities, current_priorities, SNRs, C_opt, 1)
        print("Priorities vector", priorities)

        lora.find_optimal_c_13()
        lora.C = lora.C_opt
        print(lora.compute_network_performance())

        # represent_sfs_crs(lora.C[1 + 12 * 3: 1 + 12 * 4, 0].reshape(12, 1),
        #                   save_path='lora_varying_priority_{:.2f}.png'.format(p), show=True)

        represent_sfs(lora.C[1 + 12 * 3: 1 + 12 * 4, 0].reshape(12, 1), title="Spreading Factor allocation of node i", save_path='lora_varying_priority_{:.2f}.png'.format(p), show=False)
        represent_sfs(lora.C[1 + 12 * 3: 1 + 12 * 4, 1:], title="Spreading Factor allocation of the rest of nodes", save_path='lora_varying_priority_alls_{:.2f}.png'.format(p), show=False)

    exit(-1)

if __name__ == '__main__':
    from es import PEPG

    experiment_vary_importance(num_nodos=40)
    # experiment_add_remove(num_nodos=40)


    np.random.seed(7)
    num_nodos = 10  # en lugar de 60 nodos con DC = 1% -> 6 con DC = 0.1%?

    min_rate = 1
    max_rate = 900  # vamos a modelar el T medio entre paquetes, y la lambda como inversa de esto
    lambdas_ = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, num_nodos) + min_rate)
    lengths = np.random.randint(15, 30, (lambdas_.shape[0]))

    min_priority = 0
    max_priority = 10000
    priorities = (max_priority - min_priority) * np.random.random(num_nodos) + min_priority

    min_current_priority = 1
    max_current_priority = 10000
    current_priorities = (max_current_priority - min_current_priority) * np.random.random(
        num_nodos) + min_current_priority

    warnings.warn('Recordar subir SNR')
    SNRs = np.random.randint(-23, 5, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    C_opt = np.zeros((49, lambdas_.shape[0]))
    lora = LoRaWorld(lambdas_, lengths, priorities, current_priorities, SNRs, C_opt, 1)

    POPULATION = 100
    solver = PEPG(13 * lora.N,  # number of model parameters
                  sigma_init=1,  # initial standard deviation
                  learning_rate=1e-3,  # learning rate for standard deviation
                  elite_ratio=0.15,
                  popsize=POPULATION,  # population size
                  average_baseline=True,  # set baseline to average of batch
                  weight_decay=0.00,  # weight decay coefficient
                  rank_fitness=False,  # use rank rather than fitness numbers
                  forget_best=False)  # don't keep the historical best solution)

    t1 = time.time()
    lora.find_optimal_c_13()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora.C = lora.C_opt
    print(lora.compute_network_performance())
    represent_sfs(lora.C[1 + 12 * 3: 1 + 12 * 4, :])

    mu = np.zeros((13, lora.N))
    mu[0, :] = lora.C_opt[0, :]
    mu[1:, :] = lora.C[1 + 12 * 3: 1 + 12 * 4, :]
    mu = mu.flatten()

    solver.set_mu(mu)
    t1 = time.time()
    C = lora.find_optimal_c_13_ES(solver, POPULATION, iters=5000)
    print('Elapsed computing with ES {}'.format(time.time() - t1))
    lora.C = lora.C_opt
    print(lora.compute_network_performance())

    t1 = time.time()
    print("Otra pasada con Scipy")
    lora.find_optimal_c_13(passes=1, initial=C)
    print('Elapsed computing with Scipy 2 {}'.format(time.time() - t1))

    exit(-1)
    # lora.find_genetic_optimal()

    # prob = pg.problem(my_constrained_udp(num_nodos, lora))
    # prob.c_tol = [1E-1] * (num_nodos)
    # print(prob)
    # # algo = pg.algorithm(uda=pg.nlopt('slsqp'))
    # algo = pg.algorithm(uda=pg.nlopt('auglag'))
    # algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
    # algo.set_verbosity(200)  # in this case this correspond to logs each 200 objevals
    #
    # archi = pg.archipelago(n=8, algo=algo, prob=prob, pop_size=70)
    # archi.evolve()
    # archi.wait()
    # archi.get_champions_f()
    #
    # # pop = pg.population(prob=prob, size=1)
    # # pop.problem.c_tol = [1E-1] * (num_nodos)
    # # pop = algo.evolve(pop)
    # # print(pop.get_fevals())
    # # print(pop.get_gevals())
    # exit()

    warnings.warn('Alpha = 1, solo considerando throughput')
    # lora = LoRaWorld(lambdas_, lengths, priorities, current_priorities, SNRs, C_opt, 1)
    # lora.reset(gen_random=True)
    # r1 = lora.compute_network_performance()
    # lora.find_optimal_by_adr()
    # lora.C = lora.C_opt
    # r2 = lora.compute_network_performance()

    lora = LoRaWorld(lambdas_, lengths, priorities, current_priorities, SNRs, C_opt, 1)
    lora.reset()
    print(lora.compute_network_performance())

    lora.find_optimal_c()
    lora.C = lora.C_opt
    print(lora.compute_network_performance())

    lora.find_optimal_c_13()
    lora.C = lora.C_opt
    print(lora.compute_network_performance())

    t0 = time.time()
    its = int(10000)
    l = list()
    for _ in range(its):
        # lora.get_transmittable(0)
        # lambdas_effective = [lora.get_effective_lambda(n) for n in range(lora.lambdas_.shape[0])]
        c = np.random.rand(49, lora.N)
        c = c / c.sum(axis=0)
        lora.C = c
        a = lora.compute_network_performance()  # el performance de este es siempre peor
        b = lora.compute_network_performance_precise()
        print(a, b)
        l.append(b - a)
        assert a < b
    t1 = time.time()
    print("Elapsed {} seconds per run".format((t1 - t0) / its))
    print('ok')
    from matplotlib import pyplot as plt

    plt.plot(l)
    plt.show()
    plt.hist(l)
    plt.show()
