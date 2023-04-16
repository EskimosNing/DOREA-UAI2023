from numpy.lib.function_base import append
from agent import Agent
from utils import get_utility
import random
import math
import copy


class RandomAgent(Agent):
    def __init__(self, max_round, name="Random agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)

    def reset(self):
        super().reset()

    def concess(self, type="normal"):
        self.s = random.random()

    def receive(self, last_action=None):
        super().receive(last_action)

    def gen_offer(self, offer_type = "oppo_prefer"):
        if self.domain_type == "REAL":
            if offer_type == "random": 
                offer = [random.random() for i in range(self.issue_num)]
                utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                self.utility_proposed.append(utility)
                self.offer_proposed.append(offer)
                return offer
            elif offer_type == "oppo_prefer": # using a way to esimate oppo prefer , then generate offer based on this esitimated oppo prefer
                res_offer = None
                oppo_utility = -1
                for _ in range(30):
                    offer = [random.random() for i in range(self.issue_num)]
                    utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                    tmp = get_utility(offer, self.oppo_prefer, 1 - self.condition, self.domain_type, self.oppo_issue_value)
                    if oppo_utility < tmp:
                        oppo_utility = tmp
                        res_offer = offer
                utility = get_utility(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                self.utility_proposed.append(utility)
                self.offer_proposed.append(res_offer)
                return res_offer
        elif self.domain_type == "DISCRETE":
            if offer_type == "random":
                offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                self.utility_proposed.append(utility)
                self.offer_proposed.append(offer)
                return offer
            elif offer_type == "oppo_prefer": # using a way to esimate oppo prefer , then generate offer based on this esitimated oppo prefer
                res_offer = None
                oppo_utility = -1                
                for _ in range(30):
                    offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                    utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                    tmp = get_utility(offer, self.oppo_prefer, 1 - self.condition, self.domain_type, self.oppo_issue_value)
                    if oppo_utility < tmp:
                        oppo_utility = tmp
                        res_offer = offer
                utility = get_utility(res_offer, self.prefer, self.condition, self.domain_type, self.issue_value)
                self.utility_proposed.append(utility)
                self.offer_proposed.append(res_offer)
                return res_offer