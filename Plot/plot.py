
import matplotlib.pyplot as plt
import numpy as np

double_dqn_f = "DoubleDQN-CartPole-v1-5000-episodes-batchsize-32"
dqn_f = "DQN-CartPole-v1-5000-episodes-batchsize-32"
dueling_dqn_f = "DuelingDQN-CartPole-v1-5000-episodes-batchsize-32"
noisy_dqn_f = "NoisyDQN-CartPole-v1-5000-episodes-batchsize-32"
per_dqn_f = "DQNwPER-CartPole-v1-5000-episodes-batchsize-32"
nddp_dqn_f = "NoisyDuelingDoubleDQNwPER-CartPole-v1-5000-episodes-batchsize-32"
ndp_dqn_f = "NoisyDoubleDQNwPER-CartPole-v1-5000-episodes-batchsize-32"

dqn = open(dqn_f, 'r')
double_dqn = open(double_dqn_f,'r')
dueling_dqn = open(dueling_dqn_f, 'r')
noisy_dqn = open(noisy_dqn_f, 'r')
per_dqn = open(per_dqn_f, 'r')
nddp_dqn = open(nddp_dqn_f, 'r')
ndp_dqn = open(ndp_dqn_f, 'r')

e = []
dqn_r = []
dbdqn_r = []
ddqn_r = []
ndqn_r = []
pdqn_r = []
nddpdqn_r = []
ndpdqn_r = []

for dqn, dbdqn, ddqn, ndqn, pdqn, nddpdqn, ndpdqn in zip(dqn.readlines(), double_dqn.readlines(),
                                                         dueling_dqn.readlines(), noisy_dqn.readlines(),
                                                         per_dqn.readlines(), nddp_dqn.readlines(), ndp_dqn.readlines()):
    dqn_str = dqn.split(" ")
    dbdqn_str = dbdqn.split(" ")
    ddqn_str = ddqn.split(" ")
    ndqn_str = ndqn.split(" ")
    pdqn_str = pdqn.split(" ")
    nddpdqn_str = nddpdqn.split(" ")
    ndpdqn_str = ndpdqn.split(" ")
    e.append(int(dqn_str[0]))
    dqn_r.append(int(dqn_str[2]))
    dbdqn_r.append(int(dbdqn_str[2]))
    ddqn_r.append(int(ddqn_str[2]))
    ndqn_r.append((int(ndqn_str[2])))
    pdqn_r.append((int(pdqn_str[2])))
    nddpdqn_r.append((int(nddpdqn_str[2])))
    ndpdqn_r.append((int(ndpdqn_str[2])))

# plt.plot(e, dqn_r, '-')
# plt.savefig(dqn_f+".png")
# plt.close()
#
# plt.plot(e, dbdqn_r, '-')
# plt.savefig(double_dqn_f+".png")
# plt.close()
#
# plt.plot(e, ddqn_r, '-')
# plt.savefig(dueling_dqn_f+".png")
# plt.close()
#
# plt.plot(e, ndqn_r, '-')
# plt.savefig(noisy_dqn_f+".png")
# plt.close()
#
#
# plt.plot(e, pdqn_r, '-')
# plt.savefig(per_dqn_f+".png")
# plt.close()
#
# plt.plot(e, nddpdqn_r, '-')
# plt.savefig(nddp_dqn_f+".png")
# plt.close()
#
# plt.plot(e, ndpdqn_r, '-')
# plt.savefig(ndp_dqn_f+".png")
# plt.close()

dqn_r = np.array(dqn_r)
dbdqn_r = np.array(dbdqn_r)
ddqn_r = np.array(ddqn_r)
ndqn_r = np.array(ndqn_r)
pdqn_r = np.array(pdqn_r)
nddpdqn_r = np.array(nddpdqn_r)
ndpdqn_r = np.array(ndpdqn_r)

xdqn = []
xdbdqn = []
xddqn = []
xndqn = []
xpdqn = []
xnddpdqn = []
xndpdqn = []
x = []
count = 1

for i in range(len(e)):
    if i % 200 == 0 and i!=0:
        xdqn.append(np.mean(dqn_r[i-50:i]))
        xdbdqn.append(np.mean(dbdqn_r[i-50:i]))
        xddqn.append(np.mean(ddqn_r[i-50:i]))
        xndqn.append(np.mean(ndqn_r[i-50:i]))
        xpdqn.append(np.mean(pdqn_r[i-50:i]))
        xnddpdqn.append(np.mean(nddpdqn_r[i-50:i]))
        xndpdqn.append(np.mean(ndpdqn_r[i-50:i]))
        count += 200
        x.append(count)


plt.plot(x, xdqn, '-')
plt.plot(x, xdbdqn, '-')
plt.plot(x, xddqn, '-')
plt.plot(x, xndqn, '-')
plt.plot(x, xpdqn, '-')
plt.plot(x, xnddpdqn, '-')
plt.plot(x, xndpdqn, '-')
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend(["DQN","Double DQN","Dueling DQN","Noisy DQN", "Priority Experience Replay DQN","Noisy Dueling Duoble DQN with PER", "Noisy Duoble DQN with PER"])
plt.show()

