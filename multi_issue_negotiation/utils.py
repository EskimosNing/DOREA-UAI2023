import numpy as np
from typing import Dict, List, Tuple
import collections
import math
import struct

def floatToBits(x):
    rep = struct.pack('>f', x)
    numeric = struct.unpack('>I', rep)[0]
    ret = '%x' %numeric
    ret = int(ret, 16)
    return ret

def get_utility(offer, prefer, condition, domain_type="DISCRETE", issue_value=None):  # condition 0 prefer small value, condition 1 prefer big value
    if domain_type == "DISCRETE" and issue_value == None:
        print("get_utility() need issue_value arg when domain is DISCRETE")
        exit(-1)
    utility = 0
    if offer is None:
        return utility    

    if domain_type == "REAL":
        if all(offer == np.array([-1, -1, -1])):
            return utility

        for i in range(len(offer)):
            value = (1 - condition) + (2 * condition - 1) * offer[i]
            utility += value * prefer[i]
        return utility

    elif domain_type == "DISCRETE":
        # print("in get_utility: issue_value", issue_value)
        # print("in get_utility: offer", offer)
        for i in range(len(offer)):
            value = issue_value[i][offer[i]]
            utility += value * prefer[i]
        return utility


def get_utility_with_discount(offer, prefer, condition, domain_type="DISCRETE", issue_value=None, time=1.1, discount=1.1):
    if time < 0 or time > 1 or discount < 0 or discount > 1:
        print("Exception in get_utility_with_discount()")
        exit(-1)
    utility = get_utility(offer, prefer, condition, domain_type, issue_value)
    discountedUtil = utility * math.pow(discount, time)
    return discountedUtil


def random_roll(index,prob):
    prob = prob/sum(prob)
    index = np.random.choice(index, p=prob.ravel())
    return index


class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        # if data[3] > 0:
        #     print("transtition",data)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
