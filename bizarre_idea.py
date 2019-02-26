import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pickle
from gym_lora_faster import LoRaWorld
import random
import sys
# import tensorflow as tf
import os
import time
import glob
import multiprocessing
sys.path.append('estool/')
from es import CMAES, PEPG, OpenES
import warnings
import platform
from sklearn.neural_network import MLPClassifier

global lambdas_, lengths, priorities, priorities_power, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, starting_point, suboptimal

if platform.node() == "alioth" or platform.node().startswith('ip-'):
    print('Alioth config')
    WORKERS = multiprocessing.cpu_count()   # greedy
    BASE_POP = 33
    NPOPULATION = BASE_POP if WORKERS < BASE_POP else ((WORKERS // 2) * 2 - 1)
    REPS = 32
    TEST_BASELINE = False
elif platform.node() == "xps": #  or platform.node() == "iMac-de-Ruben.local"
    print('Laptop config')
    WORKERS = multiprocessing.cpu_count()
    BASE_POP = 33
    NPOPULATION = BASE_POP if WORKERS < BASE_POP else ((WORKERS // 2) * 2 - 1)
    REPS = 32
    TEST_BASELINE = False
else:
    print('Sebas config')
    WORKERS = multiprocessing.cpu_count()
    BASE_POP = 33
    NPOPULATION = BASE_POP if WORKERS < BASE_POP else ((WORKERS // 2) * 2 - 1)
    REPS = 32
    TEST_BASELINE = False


def pretrain_network():
    def gen_sample(n):
        for _ in range(int(n)):
            # TDC = np.array([np.random.rand() * 36])
            # rareness = np.random.rand()
            # efficiency = np.random.rand()
            # consumption = np.random.rand()
            # priority = np.random.rand()
            # X = np.array([1, rareness, efficiency, TDC, consumption, priority])


            # return np.array([self.lambdas_[generated_packet], pos_lambda,
            #                  self.efficiencies[generated_packet], pos_efficiency,
            #                  self.reporting_consumption[generated_packet], pos_consumption,
            #                  TDC,
            #                  pos_tx_priority,
            #                  pos_power_priority
            #                  ])


            X = np.array([
                1,
                (1 / ((900 - 0.5) * np.random.beta(1, 5, 1) + 0.5))[0], # lambda
                np.random.rand(),   # lambda_rel
                np.random.rand() * 8 - 4,   # DR
                np.random.rand(), # DR_rel
                4 * np.random.rand(),   # C
                np.random.rand(),   # C_rel
                np.random.rand(),   # TDC_rel
                np.random.rand(),   # Priority_tx_rel
                np.random.rand()   # Priority_power_rel
            ])

            yield X

    if os.path.isfile('starting_point.p'):
        starting_point = pickle.load(open('starting_point.p', 'rb'))
        return starting_point

    clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=0.5, hidden_layer_sizes = (h_1_size, h_2_size), random_state = 1)

    p = 0.9
    X = np.array(list(gen_sample(1000)))
    y = np.random.choice([0, 1], size=1000, p=[1 - p, p]) # np.ones(int(100))
    clf.fit(X, y)
    starting_point = np.concatenate([np.concatenate([clf.coefs_[n].flatten(), clf.intercepts_[n].flatten()]) for n in range(3)])
    pickle.dump(starting_point, open('starting_point.p', 'wb'))
    print('starting point dumped')
    return starting_point


def predict(x):
    return np.argmax(np.random.multinomial(1, x))


def sample_action_simple(X, weights):
    v = np.tanh(weights.T.dot(X))
    return predict([v, 1 - v])


def sample_action(X, weights):
    weights_1 = weights[:weights_1_n]
    weights_1.shape = (D, h_1_size)
    bias_1 = weights[weights_1_n:weights_1_n + bias_1_n]
    weights_2 = weights[weights_1_n + bias_1_n:weights_1_n + bias_1_n + weights_2_n]
    weights_2.shape = (h_1_size, h_2_size)
    bias_2 = weights[weights_1_n + bias_1_n + weights_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n]
    weights_3 = weights[
                weights_1_n + bias_1_n + weights_2_n + bias_2_n:weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n]
    weights_3.shape = (h_2_size, K)
    bias_3 = weights[weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n:]

    # assert weights.shape[0] == (weights_1_n + bias_1_n + weights_2_n + bias_2_n + weights_3_n + bias_3_n)
    # assert X.shape[0] == D

    z_1 = np.tanh(weights_1.T.dot(X) + bias_1)
    z_2 = np.tanh(weights_2.T.dot(z_1) + bias_2)
    Y = 1 / (1 + np.exp(-(weights_3.T.dot(z_2) + bias_3)))
    action = predict([Y[0], 1 - Y[0]])
    return action


def rollout(env, model, std_perturbance=0, force_one=False, suboptimal=False):
    if suboptimal:
        observation = env.reset(gen_random=True)
    else:
        observation = env.reset(gen_random=False)

    observation = env.reset()
    max_time = 3600 * 10  # 24 * 2

    total_reward = 0
    info_time = 0
    done = False

    while (info_time < max_time) and not done:
        if force_one:
            action = 1
        else:
            observation = env.fixed_length_input(observation, extended=True)  # type: np.ndarray
            observation += np.random.normal(loc=0, scale=std_perturbance, size=(observation.shape[0]))
            observation = np.concatenate([[1], observation])
            action = sample_action(observation, model)

        observation, reward, done, info_time = env.step(action)
        total_reward += reward

    total_reward += env.gather_last_rewards(til=max_time + 3600)

    return total_reward, done


def rollout_rep(params):
    global lambdas_, lengths, priorities, priorities_power, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, REPS, suboptimal
    model, seed = params
    random.seed(seed)
    np.random.seed(seed)
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)

    rs = list()
    for rep in range(REPS):
        # t1 = time.time()
        r, done = rollout(env, model, suboptimal=suboptimal)
        # print('Done @', time.time() - t1, 'model', model[0])
        rs.append(r)
    rs = np.array(rs)

    # print('Finished all reps model', model[0])
    # if TEST_BASELINE:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #
    #     env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)
    #
    #     rs_baseline = list()
    #     for rep in range(REPS):
    #         r, done = rollout(env, model, force_one=True)
    #         rs_baseline.append(r)
    #     rs_baseline = np.array(rs_baseline)
    #
    #     rs -= rs_baseline

    return rs


def test_solver(solver):
    # warnings.warn('{} reps solo'.format(REPS))

    for j in range(MAX_ITERATION):
        solutions = solver.ask()

        # seeds = np.random.randint(0, 4294967295, solutions.shape[0])
        seeds = np.full((solutions.shape[0]), j)

        if WORKERS is None or WORKERS > 1:
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                p = executor.map(rollout_rep, zip(solutions, seeds))
                fitness_list = np.array(list(p))
        else:
            fitness_list = list()
            for i in range(solutions.shape[0]):
                fitness_list.append(rollout_rep([solutions[i], seeds[i]]))  # rollout(environment, solutions[i], D, K, h_1_size)

        fitness_list_means = []
        # warnings.warn('Me estoy quedando solo con REPS reps')
        for fl in fitness_list:
            # v = np.mean(np.sort(fl)[:int(np.ceil(REPS * TOP_PERCENTIL))])
            v = np.mean(fl)
            fitness_list_means.append(v)

        solver.tell(fitness_list_means)
        result = solver.result()  # first element is the best solution, second element is the best fitness

        if (j + 1) % 1 == 0:
            print("max fitness at iteration", (j + 1), result[1])

            # print("Average fitness of top 25% at iteration", (j + 1),
            #       np.sort(fitness_list_means)[-int(NPOPULATION / 4):].mean())
            print("Max fitness of this iter: {}".format(np.max(fitness_list_means)))

            timestr = time.strftime("%Y%m%d-%H%M%S")
            pickle.dump(result[0], open('models/model-{}.p'.format(timestr), 'wb'))
            # saver.save(session, os.getcwd() + '/my_test_model')

        # xs = np.concatenate([np.tile(v, (REPS, 1)) for v in solutions])
        # ys = fitness_list.flatten()
        # assert xs.shape[0] == ys.shape[0]
        #
        # for _ in range(100):  # bagging with replacement
        #     idxs = np.random.randint(0, xs.shape[0], int(xs.shape[0] * 1))
        #     solutions_ = np.array(xs[idxs])
        #     fitness_list_ = np.array(ys[idxs])
        #     pmodel.partial_fit(solutions_, fitness_list_)
        #
        # max_idxs = np.argsort(fitness_list_means)[-int(REFINED_PROP):]
        # next_solutions = list()
        # for sol_idx in max_idxs:
        #     base_solution = solutions[sol_idx]
        #     r = pmodel.boost(base_solution)[0][0]
        #     r *= 1e-3
        #     next_solutions.append(base_solution + r)

        print('it done')


if __name__ == '__main__':
    global lambdas_, lengths, priorities, priorities_power, alpha, SNRs, C_opt, D, K, h_1_size, h_2_size, MAX_ITERATION, starting_point, suboptimal

    random.seed(0)
    np.random.seed(0)
    num_nodos = 60  # en lugar de 60 nodos con DC = 1% -> 6 con DC = 0.1%?
    sub_optimal_C = False

    min_rate = 0.5
    max_rate = 100  # vamos a modelar el T medio entre paquetes, y la lambda como inversa de esto
    lambdas_ = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, num_nodos) + min_rate)
    lengths = np.random.randint(15, 30, (lambdas_.shape[0]))

    min_priority = 0
    max_priority = 1 # mod 10000
    priorities = (max_priority - min_priority) * np.random.random(num_nodos) + min_priority

    min_current_priority = 0 # mod 1
    max_current_priority = 1 # mod 10000
    priorities_power = (max_current_priority - min_current_priority) * np.random.random(
        num_nodos) + min_current_priority

    SNRs = np.random.randint(-23, 0, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    C_opt = np.zeros((49, lambdas_.shape[0]))
    alpha = 1
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)
    if sub_optimal_C:
        suboptimal = True
        env.find_optimal_by_adr()
        print('Suboptimal configuration under use')
    else:
        suboptimal = False
        if os.path.exists('c_opt_{}.p'.format(num_nodos)):
            C_opt_t = pickle.load(open('c_opt_{}.p'.format(num_nodos), 'rb'))
            if C_opt_t.shape == (49, lambdas_.shape[0]):
                env.C_opt = C_opt_t
            else:
                env.find_optimal_c_13(passes=1)
                pickle.dump(env.C_opt, open('c_opt_{}.p'.format(num_nodos), 'wb'))
        else:
            POPULATION = 100
            solver = PEPG(13 * env.N,  # number of model parameters
                          sigma_init=1,  # initial standard deviation
                          learning_rate=1e-3,  # learning rate for standard deviation
                          elite_ratio=0.15,
                          popsize=POPULATION,  # population size
                          average_baseline=True,  # set baseline to average of batch
                          weight_decay=0.00,  # weight decay coefficient
                          rank_fitness=False,  # use rank rather than fitness numbers
                          forget_best=False)  # don't keep the historical best solution)

            C = env.find_optimal_c_13_ES(solver, POPULATION, iters=1000)
            env.find_optimal_c_13(passes=1, initial=C)

            # env.find_optimal_c_13(passes=1)
            pickle.dump(env.C_opt, open('c_opt_{}.p'.format(num_nodos), 'wb'))
    # exit()

    C_opt = np.array(env.C_opt)

    # model = np.array([-3.02732536,  10.70271146,  -5.02678069, -18.22820558]) # np.random.rand(D)
    D = 10  # lambdas_.shape[0] * 2 + 1  #
    K = 1
    h_1_size = 45  # 100  # 20
    h_2_size = 5  # 50  # 10
    NPARAMS = (D + 1) * h_1_size + (h_1_size + 1) * h_2_size + (h_2_size + 1) * K

    weights_1_n = D * h_1_size
    bias_1_n = h_1_size
    weights_2_n = h_1_size * h_2_size
    bias_2_n = h_2_size
    weights_3_n = h_2_size * K
    bias_3_n = K

    # warnings.warn('Loading previous model.p')


    models = glob.glob('models/model*.p')
    if len(models) > 0:
        print('Selecting last generated model')
        models.sort(key=os.path.getmtime)
        model_path = models[-1]
        print('Found model to be loaded:', model_path)
        model = pickle.load(open(model_path, 'rb'))
    else:
        print('Generating a first good-ish model')
        model = pretrain_network()

    # model = pretrain_network()
    # model = np.random.randn(NPARAMS)
    assert model.shape[0] == NPARAMS


    # np.random.seed(0)
    # random.seed(0)
    # print(rollout(env, model, force_one=True))
    # exit()

    # exit()
    # tf.set_random_seed(0)
    # pmodel = FFN(D, 1, [30, 30])
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    # session = tf.InteractiveSession()
    # session.run(init)
    # pmodel.set_session(session)

    # X = np.random.rand(1000, D)
    # y = np.random.rand(1000) * 100
    # for _ in range(1000):
    #     idxs = np.random.randint(0, y.shape[0], int(y.shape[0] * 1))
    #     solutions_ = np.array(X[idxs])
    #     fitness_list_ = np.array(y[idxs])
    #     pmodel.partial_fit(solutions_, fitness_list_)
    #
    # # saver.save(session, os.getcwd() + '/my_test_model')
    # # exit()
    #
    #
    # argmax = np.argmax(y)
    #
    # r = pmodel.boost(X[argmax])[0][0]
    # r *= 1e-3
    #
    # est_prev = pmodel.predict(X[argmax])[0][0]
    # test_point = X[argmax] + r
    # est_post = pmodel.predict(test_point)[0][0]
    #
    # print(est_prev)
    # print(est_post)
    #
    # exit()

    # openes = OpenES(num_params=NPARAMS, popsize=NPOPULATION, antithetic=True)
    # openes.set_mu(model)

    # cmaes = CMAES(D,
    #               popsize=NPOPULATION,
    #               sigma_init=1,
    #               x0=model,
    #               weight_decay=0
    #               )

    pepg = PEPG(NPARAMS,  # number of model parameters
                sigma_init=1.00,  # initial standard deviation
                learning_rate=1e-3,  # learning rate for standard deviation
                elite_ratio=0.15,
                popsize=NPOPULATION,  # population size
                average_baseline=False,  # set baseline to average of batch
                weight_decay=0.00,  # weight decay coefficient
                rank_fitness=False,  # use rank rather than fitness numbers
                forget_best=False)  # don't keep the historical best solution)

    pepg.set_mu(model)

    MAX_ITERATION = 5000
    test_solver(pepg)




    # pmodel.partial_fit(X, y)
