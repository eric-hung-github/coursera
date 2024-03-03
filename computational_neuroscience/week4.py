import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

# q1.


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)


print(entropy(0.1))

# q2.


def mutual_information(r_with_s, r_without_s, p_s):
    p_r_and_s = r_with_s * p_s
    p_not_r_and_s = (1 - r_with_s) * p_s
    p_r_and_not_s = r_without_s * (1 - p_s)
    p_not_r_and_not_s = (1 - r_without_s) * (1 - p_s)
    p_r = p_r_and_s + p_r_and_not_s
    mi = p_r_and_s * math.log(p_r_and_s / (p_r * p_s), 2) + p_not_r_and_s * math.log(p_not_r_and_s / ((1 - p_r) * p_s), 2) + p_r_and_not_s * math.log(p_r_and_not_s / (p_r * (1 - p_s)), 2) + p_not_r_and_not_s * math.log(p_not_r_and_not_s / ((1 - p_r) * (1 - p_s)), 2)
    return mi


print(mutual_information(1/2, 1/18, 0.1))

# q6.
# with open('week4_q6.pickle', 'rb') as f6:
#     tuning = pickle.load(f6)

#     plt.figure(figsize=(10, 5))

#     for key,value in tuning.items():
#         if key.startswith("stim"):
#             continue
#         plt.plot(tuning["stim"], np.mean(tuning[key],axis=0), label=key)

#     plt.title('Tuning curve')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# q7.
# with open('week4_q6.pickle', 'rb') as f6:
#     tuning = pickle.load(f6)

#     plt.figure(figsize=(10, 5))

#     for key,value in tuning.items():
#         if key.startswith("stim"):
#             continue
#         plt.scatter(np.mean(tuning[key],axis=0)/15, np.var(tuning[key],axis=0), label=key)

#     plt.title('Tuning curve')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# q8.
with open('week4_q6.pickle', 'rb') as f6:
    with open('week4_q8.pickle', 'rb') as f8:
        tuning_data = pickle.load(f6)
        pop_coding = pickle.load(f8)

        # caculate maximum average firing rate (over any of the stimulus values in 'tuning') for a neuron as the value of r_max
        r_max = []
        for i in range(4):
            r_max.append(np.max(np.mean(tuning_data["neuron"+str(i+1)], axis=0)))

        # normalized the vectors r1 to r4 by dividing the each vector by corresponding rmax value and named the vectors r1norm to r4 norm
        r = []
        for i in range(4):
            r.append(pop_coding["r"+str(i+1)])
        r_norm = []
        for i in range(4):
            r_norm.append(r[i] / r_max[i])
        r_norm = np.array(r_norm)

        # calculated mean value in each vector from r1norm to r4norm
        r_mean = np.mean(r_norm, axis=1)

        # multiplied r_mean with corresponding c1 to c4 vectors
        c = []
        for i in range(4):
            c.append(pop_coding["c"+str(i+1)])
        c = np.array(c)
        v = np.sum(r_mean[:, np.newaxis] * c, axis=0)

        #  caculate arctan of v into 360
        angle = math.atan2(v[0], v[1])
        angle = math.degrees(angle)
        if angle < 0:
            angle += 360
        print(v, angle)
