from utils import get_utility, get_utility_with_discount
import random
import numpy as np
import copy
import math
from itertools import product
from functools import reduce


class Agent:
    def __init__(self, max_round, name,discount=1.0, u_max=1, u_min=0.1, issue_num=3):
        self.max_round = max_round
        self.issue_num = issue_num
        self.u_max = u_max
        self.u_min = u_min
        self.name = name
        self.oppo_prefer = [1] * self.issue_num
        self.accept = False
        self.terminate = False
        self.t = 0
        self.relative_t = 0

        # for discrete domain
        self.domain_type = "REAL"
        self.issue_name = []  
        self.issue_value = None  
        self.oppo_issue_value = None
        self.oppo_issue_value_not_normalized = None
        self.bidSpace = 0
        self.bestBid = None
        self.issue_weight = None
        self.allBids = None

        # for discount factor and reservation value
        self.discount = 1.0
        self.reservation = self.u_min

        # for update oppo_issue_value estimator
        self.oppo_gamma = 0.4
        self.oppo_lambda = 0.5


    def reset(self):  
        self.offer_received = []  # the list contains the utility of the offer received
        self.offer_proposed = []  # the list contains the utility of the offer proposed
        self.utility_received = []
        self.utility_proposed = []
        self.value_received = []  
        self.s = self.u_max
        self.t = 0
        self.relative_t = 0
        self.accept = False
        self.terminate = False
        self.oppo_prefer = [1] * self.issue_num
        self.allBids = None  
        # validateDiscount
        if self.discount <= 0 or self.discount > 1:
            self.discount = 1
        if self.domain_type == "REAL":
            self.bestBid = [self.condition] * self.issue_num
        if self.domain_type == "DISCRETE":
            
            self.oppo_issue_value = copy.deepcopy(self.issue_value)
            for i in range(self.issue_num):
                for j in self.issue_value[i].keys():
                    self.oppo_issue_value[i][j] = 1
            self.oppo_issue_value_not_normalized = copy.deepcopy(self.oppo_issue_value)


    def set_prefer(self, prefer):
        self.prefer = prefer

    def set_condition(self, condition):
        self.condition = condition

    def set_t(self, t):
        self.t = t
        self.relative_t = self.t / self.max_round
    def get_relative_t(self):
        return self.relative_t

    def get_discount(self):
        return self.discount

    def get_t(self):
        return self.t

    def concess(self, type="normal"):
        pass

    def receive(self, last_action=None):
        if last_action is not None:
            offer = last_action
            utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_received.append(utility)
            self.offer_received.append(offer)
            if self.domain_type == "DISCRETE":
                value_recv = []
                for i in range(len(offer)):
                    value_recv.append(self.issue_value[i][offer[i]])
                self.value_received.append(value_recv)
            self.update_oppo_issue_value_estimation(offer, self.relative_t)
            self.oppo_prefer_estimater()
            self.concess()
            self.accept = (utility >= self.s and self.t <= self.max_round)


    def update_oppo_issue_value_estimation(self, bid, relative_t):
        if self.domain_type == "DISCRETE":
            
            for i in range(self.issue_num):
                value = bid[i]
                self.oppo_issue_value_not_normalized[i][value] += math.pow(self.oppo_gamma, relative_t)
            
            
            for i in range(self.issue_num):
                maxValue = 0.0
                for key in self.oppo_issue_value_not_normalized[i].keys():                    
                    if math.pow(self.oppo_issue_value_not_normalized[i][key], self.oppo_lambda) > maxValue:
                        maxValue = math.pow(self.oppo_issue_value_not_normalized[i][key], self.oppo_lambda)
                for key in self.oppo_issue_value[i].keys():
                    self.oppo_issue_value[i][key] = math.pow(self.oppo_issue_value_not_normalized[i][key], self.oppo_lambda) / maxValue


    def oppo_prefer_estimater(self):
        if self.domain_type == "REAL":
            tmp = np.array(self.offer_received)
            deviation = np.std(tmp, axis=0)
            # relative_oppo_prefer = [deviation[1] * deviation[2], deviation[0] * deviation[2], deviation[1]*deviation[0]]
            relative_oppo_prefer = []
            for i in range(self.issue_num):
                relative_oppo_prefer.append(self.cal_relative_oppo_prefer_i(deviation, i))
            if sum(relative_oppo_prefer) != 0:
                self.oppo_prefer = [i / sum(relative_oppo_prefer) for i in relative_oppo_prefer]
        elif self.domain_type == "DISCRETE":
            tmp = np.array(self.value_received)
            deviation = np.std(tmp, axis=0)
            relative_oppo_prefer = []
            for i in range(self.issue_num):
                relative_oppo_prefer.append(self.cal_relative_oppo_prefer_i(deviation, i))
            if sum(relative_oppo_prefer) != 0:
                self.oppo_prefer = [i / sum(relative_oppo_prefer) for i in relative_oppo_prefer]

    def cal_relative_oppo_prefer_i(self, deviation, i):
        ret = 1
        for j in range(self.issue_num):
            if j != i:
                ret *= deviation[j]
        return ret

    def act(self):
        return self.gen_offer()


    def gen_offer(self, offer_type="nearest"):
        if self.domain_type == "REAL":
            # print("REAL domain gen_offer")
            if offer_type == "random":
                if self.s > 0.95:
                    offer = [self.condition] * self.issue_num
                    self.utility_proposed.append(1)
                    self.offer_proposed.append(offer)
                    return offer
                else:
                    while True:
                        offer = [random.random() for i in range(self.issue_num)]
                        utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                        #utility = get_discount)(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                        if utility >= self.s and utility <= self.s + 0.05:                            
                            self.utility_proposed.append(utility)
                            self.offer_proposed.append(offer)
                            return offer
            elif offer_type == "oppo_prefer": # using a way to esimate oppo prefer , then generate offer based on this esitimated oppo prefer
                if self.s > 0.95:
                    offer = [self.condition] * self.issue_num
                    self.utility_proposed.append(1)
                    self.offer_proposed.append(offer)
                    return offer
                else:
                    res_offer = None
                    oppo_utility = -1
                    for _ in range(30):
                        while True:
                            offer = [random.random() for i in range(self.issue_num)]
                            utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                            if utility >= self.s and utility <= self.s + 0.05:
                                tmp = get_utility(offer, self.oppo_prefer, 1 - self.condition, self.domain_type, self.oppo_issue_value)
                                if oppo_utility < tmp:
                                    oppo_utility = tmp
                                    res_offer = offer
                                break
                    utility = get_utility(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                    self.utility_proposed.append(utility)
                    self.offer_proposed.append(res_offer)
                    return res_offer
        elif self.domain_type == "DISCRETE":
            if offer_type == "random": 
                if self.s > 0.95:  
                    self.utility_proposed.append(1)
                    self.offer_proposed.append(self.bestBid) 
                    return self.bestBid
                else:      
                    for _ in range(self.bidSpace * 10):
                        offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                        utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                        if utility >= self.s and utility <= self.s + 0.05:
                            # self.utility_proposed.append(utility)
                            self.utility_proposed.append(self.s.item())
                            self.offer_proposed.append(offer)
                            return offer
                    if len(self.offer_proposed) == 0:
                        offer = self.bestBid
                        utility = 1
                    else:
                        offer = self.offer_proposed[-1]
                        utility = self.utility_proposed[-1]
                    # to reduce the uncertainty of the network, make the following change:
                    # self.utility_proposed.append(utility)
                    self.utility_proposed.append(self.s.item())
                    self.offer_proposed.append(offer)
                    return offer
            elif offer_type == "oppo_prefer": # using a way to esimate oppo prefer , then generate offer based on this esitimated oppo prefer
                if self.s > 0.95:
                    self.utility_proposed.append(1)
                    self.offer_proposed.append(self.bestBid)
                    return self.bestBid
                else:
                    res_offer = None
                    oppo_utility = -1
                    for _ in range(10):
                        for _ in range(self.bidSpace * 8):
                            offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                            utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                            if utility >= self.s and utility <= self.s + 0.05:
                                tmp = get_utility(offer, self.oppo_prefer, 1 - self.condition, self.domain_type, self.oppo_issue_value)
                                if oppo_utility < tmp:
                                    oppo_utility = tmp
                                    res_offer = offer
                                break
                    if res_offer is None and len(self.offer_proposed) == 0:
                        res_offer = self.bestBid
                    elif res_offer is None:
                        res_offer = self.offer_proposed[-1]
                    # utility = get_utility(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                    # self.utility_proposed.append(utility)
                    self.utility_proposed.append(self.s.item())
                    self.offer_proposed.append(res_offer)
                    return res_offer
            elif offer_type == "nearest":                
                res_offer = None
                distance = 100

                if self.allBids is None:
                    self.allBids = self.getAllBids()
                for _bid in self.allBids:
                    utility = get_utility(_bid, self.prefer, self.condition, self.domain_type, self.issue_value)
                    #utility = get_utility_with_discount(_bid, self.prefer, self.condition, self.domain_type, self.issue_value,self.relative_t,self.discount)
                    if abs(self.s - utility) < distance:
                        distance = abs(self.s - utility)
                        res_offer = _bid
                
                utility = get_utility(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                #utility = get_utility_with_discount(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value,self.relative_t,self.discount)
                self.utility_proposed.append(utility)
                self.offer_proposed.append(res_offer)
                return res_offer
                

    def lists_combination(self, lists, code=''):
            
        def myfunc(list1, list2):
            return [str(i)+code+str(j) for i in list1 for j in list2]

        return reduce(myfunc, lists)

    def getAllBids(self):
        _allBids = []
        issueValues = self.getIssueValues(issue_value_type = self.domain_type)
        resList = self.lists_combination(issueValues,'@')
        for i in range(len(resList)):
            tmpList = resList[i].split('@')
            _allBids.append(tmpList)
        return _allBids


    def getIssueValues(self, issue_value_type="REAL"):
        retvals = []
        if issue_value_type == "DISCRETE":            
            for i in range(self.issue_num):
                retvals.append(list(self.issue_value[i].keys()))
        elif issue_value_type == "INTEGER":
            pass
        elif issue_value_type == "REAL":
            for i in range(self.issue_num):
                i_issue_value = []
                upperBound = 1.0
                lowerBound = 0.0
                intervalReal = (upperBound - lowerBound) / 10
                for i in range(11):
                    i_issue_value.append(lowerBound + i * intervalReal)
                retvals.append(i_issue_value)
        return retvals  # list, element i means i_th issue's all possible values,like ["dell", "lenova", "HP"]

    # this.OpponentModel.getBidEvaluation
    def getBidEvaluation(self, bid):
        ret = 0
        oppo_issue_weight = self.issue_weight
        oppo_issue_value = []
        for i in range(self.issue_num):
            oppo_issue_value.append((i + 1) / self.issue_num)
        for i in range(self.issue_num):
            ret += oppo_issue_weight[i] * oppo_issue_value[i]
        return ret

    def cal_opposition(self):
        if self.domain_type == "DISCRETE":
            minn = math.hypot(1.0, 1.0)
            _allBids = self.getAllBids()
            for _bid in _allBids:
                u1 = get_utility(_bid, self.issue_weight, self.condition, self.domain_type, self.issue_value)
                u2 = self.getBidEvaluation(_bid)
                u3 = math.hypot(1.0-u1, 1.0-u2)
                if u3 < minn:
                    minn = u3
            return minn


class AgentWithDiscreteDomain(Agent):

    def __init__(self, max_round, name, u_max, u_min, issue_num):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)


class TimeAgent(Agent):

    def __init__(self, max_round, name, beta):
        super().__init__(max_round=max_round, name=name)
        self.beta = beta

    def concess(self):
        self.s = max(self.u_min, self.u_min + (self.u_max - self.u_min) * (1 - (self.t / self.max_round) ** (1 / self.beta)))

    def reset(self):
        super().reset()
    
    def act(self):
        return self.gen_offer()


class TimeAgentBoulware(TimeAgent):
    def __init__(self, max_round, name, beta=0.25):
        super().__init__(max_round=max_round, name=name, beta=beta)


class TimeAgentLinear(TimeAgent):
    def __init__(self, max_round, name, beta=1):
        super().__init__(max_round=max_round, name=name, beta=beta)


class TimeAgentConceder(TimeAgent):
    def __init__(self, max_round, name, beta=4):
        super().__init__(max_round=max_round, name=name, beta=beta)


class BehaviorAgent(Agent):
    def __init__(self, max_round, name, delta, kind_rate):
        super().__init__(max_round=max_round, name=name)
        self.delta = delta
        self.kind_rate = kind_rate

    def concess(self):
        if len(self.utility_received) - self.delta >= 0:
            self.s = min(self.u_max, max(self.u_min, self.s + self.kind_rate * (self.utility_received[-self.delta] - self.utility_received[-self.delta+1])))

    def act(self):
        # return self.gen_offer(offer_type="random")
        return self.gen_offer()

class BehaviorAgentAverage(BehaviorAgent):
    def __init__(self, max_round, name, delta=2, kind_rate=1):
        super().__init__(max_round=max_round, name=name, delta=delta, kind_rate=kind_rate)
