from matplotlib import pyplot as plt
import pickle

NUM_TESTS = 1000
num_nodos = 50

fl1 = pickle.load(open('fitness_list_50.p', 'rb'))
fl2 = pickle.load(open('fitness_list_2_50.p', 'rb'))

# fl1 = pickle.load(open('fitness_list_sub_50.p', 'rb'))
# fl2 = pickle.load(open('fitness_list_sub_2_50.p', 'rb'))

plt.hist(fl1, bins=20, range=(min(fl1.min(), fl2.min()), max(fl1.max(), fl2.max())), fc=(1, 0, 0, 1), density=False, histtype='step')
plt.hist(fl2, bins=20, range=(min(fl1.min(), fl2.min()), max(fl1.max(), fl2.max())), fc=(0, 0, 1, 1), density=False, histtype='step')

plt.legend([r'Always Transmit policy - $C_{opt}$', r'Proposed policy - $C_{opt}$'], loc=1)
# plt.legend([r'Always Transmit policy - $C$', r'Proposed policy - $C$'], loc=1)

plt.axvline(fl1.mean(), color=(0, 0, 1, 0.8), linewidth=1, linestyle='dashed')
plt.axvline(fl2.mean(), color=(1, 0, 0, 0.8), linewidth=1, linestyle='dashed')
plt.xlabel(r'$P_{\pi}$ (bits)')
plt.xlim([5500, 6200])
plt.ylabel('Number of occurrences (out of '+str(NUM_TESTS)+')')
plt.grid(False)
plt.title('Histogram of ' + r'$P_{\pi}$' + ' for 1000 randomized networks of {} nodes'.format(num_nodos))
plt.savefig('comparison_high.png', dpi=600)
plt.savefig('Performance Copt 50 nodes.png', dpi=100)
# plt.savefig('Performance Cno opt 50 nodes.png', dpi=100)
plt.show()

