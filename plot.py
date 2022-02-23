import matplotlib.pyplot as plt

plt.close("all")
plt.figure(figsize=(6, 4))
load = [10,20,30,40,50,60]
reward={'NS':[0.749,0.749,0.749,0.749,0.749,0.749], 'AS':[0.758, 0.759, 0.769, 0.772, 0.776, 0.775], 'RS':[0.732, 0.742,0.744,0.747,0.749,0.748],'EF':[0.759,0.873,0.887,0.892,0.893,0.895,]}
for policy in ['NS', 'AS', 'RS', 'EF']:
    # reward[policy][0] = 100
    plt.plot(load, reward[policy], label=policy)
    # plt.scatter(load, reward[policy], )
    # plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], reward[policy], label=policy)

plt.xlabel('Deadline')
plt.ylabel('Accuracy')

plt.legend()
plt.grid()

plt.ylim(0.72, 0.9)
plt.savefig( 'evaluation.png', dpi=400)
plt.show()
plt.clf()