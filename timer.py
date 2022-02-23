from timeit import repeat
from selection import *

for size in (10, 500,1000,2000, 5000, 8000):
    te = []
    for j in range(1000):
        temp = 2 * np.random.randn(size, 4)
        state_t = np.zeros((output_features,))
        t = repeat('RNN_forward()', 'from __main__ import RNN_forward', number=1, repeat=1000)
        te.append(min(t))
    result[0][index] = min(te)*1000
    result[1][index] = np.mean(te) * 1000
    result[2][index] = max(te) * 1000
    index += 1
    print(str(size) + ':', '%.5f' % (min(te) * 1000), '%.5f' % (np.mean(te) * 1000), '%.5f' % (max(te) * 1000))
print('totally cost',time_end-time_start)