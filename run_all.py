import numpy as np

try:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    headless = False
except:
    headless = True

import subprocess
import os.path
import re
import random
from gym_lora_faster import LoRaWorld
import pickle
from estool.es import PEPG

nodos = [20, 30, 40, 50, 60, 70, 90, 140, 200]
seeds = {20: 0, 30: 0, 40: 0, 50: 7, 60: 8, 70: 0, 90: 0, 140: 0, 200: 0}

# nodos = [20, 30, 40, 50, 60, 70, 90, 140]
# seeds = {20: 0, 30: 0, 40: 0, 50: 7, 60: 8, 70: 0, 90: 0, 140: 0}

# LA SEMILLA DE 60 LA HE CAMBIADO A 8, (la run de 60 se está ejecutando en alioth)]

"""
Con pocos nodos lo que más impacto tiene es la configuración global de la red
Con muchos nodos lo que más impacto tiene es la política de actualización de la config. global

El efecto global es que la mejora es constante (independiente del número de nodos de la red)

Evaluar C_opt vs C_adr por separado (indica el maximum network throughput divided by number of nodes to compute an
average maximum attainable throughput per node) -> lo que comparamos es throughput (no accum throughput)

Luego evaluar AT vs ANN usando C_opt para evaluar el proceso de actualización

Presentar global con AT + C_adr vs ANN + C_opt

"""

performances_ADR = list()
performances_OPT = list()

for nodo in nodos:
    print('Computing', nodo)
    random.seed(seeds[nodo])
    np.random.seed(seeds[nodo])
    num_nodos = nodo
    sub_optimal_C = True

    min_rate = 0.5
    max_rate = 100  # vamos a modelar el T medio entre paquetes, y la lambda como inversa de esto
    lambdas_ = 1 / ((max_rate - min_rate) * np.random.beta(1, 5, num_nodos) + min_rate)
    lengths = np.random.randint(15, 30, (lambdas_.shape[0]))

    min_priority = 0
    max_priority = 1  # mod 10000
    priorities = (max_priority - min_priority) * np.random.random(num_nodos) + min_priority

    min_current_priority = 0  # mod 1
    max_current_priority = 1  # mod 10000
    priorities_power = (max_current_priority - min_current_priority) * np.random.random(
        num_nodos) + min_current_priority

    SNRs = np.random.randint(-23, 0, (num_nodos,))  # (1 - 0.95) * np.random.random(5) + 0.95

    C_opt = np.zeros((49, lambdas_.shape[0]))
    alpha = 1
    env = LoRaWorld(lambdas_, lengths, priorities, priorities_power, SNRs, C_opt, alpha)

    env.find_optimal_by_adr()
    env.C = env.C_opt
    performance = env.compute_network_performance()
    performances_ADR.append(performance)

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

        C = env.find_optimal_c_13_ES(solver, POPULATION, iters=3000)
        env.find_optimal_c_13(passes=1, initial=C)

        # env.find_optimal_c_13(passes=1)
        pickle.dump(env.C_opt, open('c_opt_{}.p'.format(num_nodos), 'wb'))

    env.C = env.C_opt
    performance = env.compute_network_performance()
    performances_OPT.append(performance)

performances_ADR = np.array(performances_ADR)
performances_OPT = np.array(performances_OPT)
diff = 100 * (performances_OPT - performances_ADR) / performances_ADR

if headless:
    pickle.dump(performances_ADR, open('performances_ADR.p', 'wb'))
    pickle.dump(performances_OPT, open('performances_OPT.p', 'wb'))
else:
    fig1, ax1 = plt.subplots()
    ax1.plot(nodos, performances_OPT, 'k--', linewidth=2)
    ax1.plot(nodos, performances_ADR, 'k-', linewidth=2)
    ax1.set_ylabel(r'$\Gamma$' + ' in bytes/s', fontsize=14)
    ax1.set_xlabel('Number of nodes in the network', fontsize=14)
    ax1.set_title('Average throughput per node vs network size', fontsize=15)
    ax1.set_ylim([0, 1.6])

    ax2 = ax1.twinx()
    ax2.plot(nodos, diff, '-.', color=(0, 0, 1, 1), linewidth=2)
    ax2.set_ylabel('Improvement', color='b', fontsize=14)
    ax2.set_ylim([0, 100])
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{0:.0f}%'.format(x) for x in vals]) # important, this line should gou AFTER set_ylim
    ax2.tick_params('y', colors='b')


    legend_elements = [
        Line2D([0], [0], color='k', lw=2, linestyle='--', label='Average throughput per node with ' + r'$C_{OPT}$'),
        Line2D([0], [0], color='k', lw=2, linestyle='-', label='Average throughput per node with ' + r'$C_{ADR}$'),
        Line2D([0], [0], color='b', lw=2, linestyle='-.', label='Difference between them')
    ]
    ax1.legend(handles=legend_elements, loc=1, fontsize=12)

    fig1.tight_layout()
    fig1.savefig('throughput_high.png', dpi=600)
    fig1.savefig('throughput_low.png', dpi=200)
    fig1.savefig('throughput.eps')
    plt.show()


# varying priority of first node
nodo = 40   # fourty nodes
seed = seeds[nodo]
c_subopt = 0
times_mean = []
times_std = []
for p in np.linspace(0, 10, 11):
    out_path = 'output_{}_varying.txt'.format(p)

    if not os.path.isfile(out_path):
        print('Running importance varying simulation for', nodo, 'nodes')
        result = subprocess.run(['python3', 'evaluate_model.py', str(seed), str(nodo), str(c_subopt), str(p)],
                                stdout=subprocess.PIPE)

        with open(out_path, 'w') as file:
            file.write(result.stdout.decode('utf-8'))

        print('End simulation for', nodo, 'nodes')

    assert os.path.isfile(out_path)
    with open(out_path, 'r') as file:
        for line in file:
            # AT mean: 65176.2609227589. AT std: 641.8253626717383
            # ANN mean: 65682.35275129596. ANN std: 221.2127640103793
            if "was updated" in line:
                matchObj = re.match(r'First node was updated with N\(([\d.]+), ([\d.]+)\) seconds', line)
                assert matchObj is not None
                mean = float(matchObj.group(1))
                std = float(matchObj.group(2))

                times_mean.append(mean)
                times_std.append(std)

exit()
# opt
means_AT_opt = []
stds_AT_opt = []
means_ANN_opt = []
stds_ANN_opt = []

c_subopt = 0
for nodo in nodos:
    seed = seeds[nodo]
    out_path = 'output_{}.txt'.format(nodo)

    if not os.path.isfile(out_path):
        print('Running simulation for', nodo, 'nodes')
        result = subprocess.run(['python3', 'evaluate_model.py', str(seed), str(nodo), str(c_subopt)],
                                stdout=subprocess.PIPE)

        with open(out_path, 'w') as file:
            file.write(result.stdout.decode('utf-8'))

        print('End simulation for', nodo, 'nodes')

    assert os.path.isfile(out_path)

    with open(out_path, 'r') as file:
        for line in file:
            # AT mean: 65176.2609227589. AT std: 641.8253626717383
            # ANN mean: 65682.35275129596. ANN std: 221.2127640103793
            if "AT mean" in line:
                matchObj = re.match(r'AT mean: ([\d]+.[\d]+). AT std: ([\d]+.[\d]+)', line)
                assert matchObj is not None
                mean = float(matchObj.group(1)) / 1e3
                std = float(matchObj.group(2)) / 1e3

                means_AT_opt.append(mean)
                stds_AT_opt.append(std)

            if "ANN mean" in line:
                matchObj = re.match(r'ANN mean: ([\d]+.[\d]+). ANN std: ([\d]+.[\d]+)', line)
                assert matchObj is not None
                mean = float(matchObj.group(1)) / 1e3
                std = float(matchObj.group(2)) / 1e3

                means_ANN_opt.append(mean)
                stds_ANN_opt.append(std)
# Plot opts
means_AT_opt = np.array(means_AT_opt)
means_ANN_opt = np.array(means_ANN_opt)
diff_opt = 100 * (means_ANN_opt - means_AT_opt) / means_AT_opt

if not headless:
    fig1, ax1 = plt.subplots()
    ax1.errorbar(nodos, means_AT_opt, yerr=stds_AT_opt, fmt='k', capsize=4, linestyle='-', linewidth=2)
    ax1.errorbar(nodos, means_ANN_opt, yerr=stds_ANN_opt, fmt='k', capsize=4, linestyle='--', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(nodos, diff_opt, 'b-.', linewidth=2)
    ax2.set_ylabel('Improvement', color='b', fontsize=14)
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{0:.0f}%'.format(x) for x in vals])
    ax2.tick_params('y', colors='b')

    legend_elements = [
        Line2D([0], [0], color='k', lw=2, linestyle='--', label='Proposed policy with ' + r'$C_{opt}$'),
        Line2D([0], [0], color='k', lw=2, linestyle='-', label='Always Update policy with ' + r'$C_{opt}$'),
        Line2D([0], [0], color='b', lw=2, linestyle='-.', label='Difference between them')
    ]
    ax1.legend(handles=legend_elements, loc=9, fontsize=12)
    ax1.grid(False)
    ax1.set_xlabel('Number of nodes in the network', fontsize=14)
    ax1.set_ylabel(r'$P_{\pi}$' +' in Kbytes', fontsize=14)
    ax1.set_title('Accumulated average throughput per node vs network size', fontsize=15)
    fig1.tight_layout()
    plt.savefig('proposed_vs_at_high.png', dpi=600)
    plt.savefig('proposed_vs_at_low.png', dpi=200)
    plt.savefig('proposed_vs_at.eps')
    plt.show()
    # fig1.savefig('results_c_opt.png', dpi=600)

# subopt
means_AT_subopt = []
stds_AT_subopt = []
means_ANN_subopt = []
stds_ANN_subopt = []

c_subopt = 1
for nodo in nodos:
    seed = seeds[nodo]
    out_path = 'output_{}_sub.txt'.format(nodo)

    if not os.path.isfile(out_path):
        print('Running simulation for', nodo, 'nodes')
        result = subprocess.run(['python3', 'evaluate_model.py', str(seed), str(nodo), str(c_subopt)],
                                stdout=subprocess.PIPE)

        with open(out_path, 'w') as file:
            file.write(result.stdout.decode('utf-8'))

        print('End simulation for', nodo, 'nodes')

    assert os.path.isfile(out_path)

    with open(out_path, 'r') as file:
        for line in file:
            # AT mean: 65176.2609227589. AT std: 641.8253626717383
            # ANN mean: 65682.35275129596. ANN std: 221.2127640103793
            if "AT mean" in line:
                matchObj = re.match(r'AT mean: ([\d]+.[\d]+). AT std: ([\d]+.[\d]+)', line)
                assert matchObj is not None
                mean = float(matchObj.group(1)) / 1e3
                std = float(matchObj.group(2)) / 1e3

                means_AT_subopt.append(mean)
                stds_AT_subopt.append(std)

            if "ANN mean" in line:
                matchObj = re.match(r'ANN mean: ([\d]+.[\d]+). ANN std: ([\d]+.[\d]+)', line)
                assert matchObj is not None
                mean = float(matchObj.group(1)) / 1e3
                std = float(matchObj.group(2)) / 1e3

                means_ANN_subopt.append(mean)
                stds_ANN_subopt.append(std)

# Plot subopts
means_AT_subopt = np.array(means_AT_subopt)
# means_ANN_subopt = np.array(means_ANN_subopt)
diff_subopt = 100 * (means_ANN_opt - means_AT_subopt) / means_AT_subopt

if not headless:
    fig1sub, ax1sub = plt.subplots()
    ax1sub.errorbar(nodos, means_AT_subopt, yerr=stds_AT_opt, fmt='k', capsize=4, linestyle='-', linewidth=2)
    ax1sub.errorbar(nodos, means_ANN_opt, yerr=stds_ANN_opt, fmt='k', capsize=4, linestyle='--', linewidth=2)
    # ax1sub.set_xlim([10, 160])
    ax2sub = ax1sub.twinx()
    ax2sub.plot(nodos, diff_subopt, 'b-.', linewidth=2)
    ax2sub.set_ylabel('Improvement', color='b')
    ax2sub.set_ylim([0, 150])
    vals = ax2sub.get_yticks()
    ax2sub.set_yticklabels(['{0:.0f}%'.format(x) for x in vals])    # important, this line should gou AFTER set_ylim
    ax2sub.tick_params('y', colors='b')

    legend_elements = [
        Line2D([0], [0], color='k', lw=2, linestyle='--', label=r'$\pi^{*}$' + ' and ' + r'$C_{opt}$'),
        Line2D([0], [0], color='k', lw=2, linestyle='-', label=r'$\pi_{au}$' + ' and ' + r'$C_{ADR}$'),
        Line2D([0], [0], color='b', lw=2, linestyle='-.', label='Difference between them')
    ]
    ax1sub.legend(bbox_to_anchor=(-0.15, 0.91, 1., .102), handles=legend_elements, loc=9)
    ax1sub.grid(False)
    ax1sub.set_xlabel('Number of nodes in the network')
    ax1sub.set_ylabel(r'$P_{\pi}$' + ' in Kbytes')
    ax1sub.set_title('Accumulated average throughput per node vs network size')
    fig1sub.tight_layout()
    plt.savefig('all_vs_none_high.png', dpi=600)
    plt.savefig('all_vs_none_low.png', dpi=200)
    plt.savefig('all_vs_none.eps')
    plt.show()
