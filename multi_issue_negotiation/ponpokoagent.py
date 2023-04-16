from agent import Agent
from utils import get_utility
import random
import numpy as np
import math

class PonPokoAgent(Agent):
    def __init__(self, max_round, name="ponpoko agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.lastReceivedBid = None  # Bid
        self.lBids = None  # List<BidInfo>
        self.threshold_low = 0.99
        self.threshold_high = 1.0
        self.PATTERN_SIZE = 5
        self.pattern = 0

    def reset(self):
        super().reset()
        self.lBids = list(AgentTool.generateRandomBids(30000, self.prefer, self.issue_num, self.issue_value))
        # print('\nbefore sort - self.lBids:', self.lBids[0].getutil())
        # print()
        self.lBids.sort(key=self.comp, reverse=True)
        # print('after sort - self.lBids:', self.lBids[0].getutil())
        # print()
        self.PATTERN_SIZE = 5
        self.pattern = random.randint(0, self.PATTERN_SIZE-1)
        self.threshold_low = 0.99
        self.threshold_high = 1.0
        self.lastReceivedBid = None

    def comp(self, bidInfo):
        return bidInfo.getutil()


    def receive(self, last_action=None):
        if last_action is not None:
            utility = get_utility(last_action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.lastReceivedBid = last_action
            self.offer_received.append(last_action)
            self.utility_received.append(utility)

    def gen_offer(self):
        time = math.ceil(self.t/2) / (self.max_round/2)
        if self.pattern == 0:
            self.threshold_high = 1 - 0.1 * time
            self.threshold_low = 1 - 0.1 * time - 0.1 * abs(math.sin(40 * time))
        elif self.pattern == 1:
            self.threshold_high = 1
            self.threshold_low = 1 - 0.22 * time
        elif self.pattern == 2:
            self.threshold_high = 1 - 0.1 * time
            self.threshold_low = 1 - 0.1 * time - 0.15 * abs(math.sin(20 * time))
        elif self.pattern == 3:
            self.threshold_high = 1 - 0.05 * time
            self.threshold_low = 1 - 0.1 * time
            if time > 0.99:
                self.threshold_low = 1 - 0.3 * time
        elif self.pattern == 4:
            self.threshold_high = 1 - 0.15 * time * abs(math.sin(20 * time))
            self.threshold_low = 1 - 0.21 * time * abs(math.sin(20 * time))
        else:
            self.threshold_high = 1 - 0.1 * time
            self.threshold_low = 1 - 0.2 * abs(math.sin(40 * time))
        
        # Accept
        if self.lastReceivedBid is not None:
            if get_utility(self.lastReceivedBid, self.prefer, 1, 'DISCRETE', self.issue_value) > self.threshold_low:
                self.accept = True
                return None
        
        bid = None
        while bid is None:
            bid = AgentTool.selectBidfromList(self.lBids, self.threshold_high, self.threshold_low)
            if bid is None:
                self.threshold_low = self.threshold_low - 0.0001
        bid_util = get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)
        self.offer_proposed.append(bid)
        self.utility_proposed.append(bid_util)
        return bid


class BidInfo:
    def __init__(self, bid, u=0.0):
        self.bid = bid
        self.util = u
    
    def setutil(self, u):
        self.util = u
    
    def getBid(self):
        return self.bid

    def getutil(self):
        return self.util

    def __hash__(self):
        if self.bid is None:
            return 0
        hashValue = 0
        for value in self.bid:
            value_len = len(value)
            h = 0
            for i in range(value_len):
                h = int(31 * h + ord(value[i]))
            hashValue = int(hashValue + h)
        return hashValue

    def __eq__(self, other):
        if isinstance(other, BidInfo):
            return self.bid == other.bid
        else:
            return False


class AgentTool:
    def selectBidfromList(bidInfoList, higerutil, lowwerutil):
        bidInfos = []
        for bidInfo in bidInfoList:                     
            if bidInfo.getutil() <= higerutil and bidInfo.getutil() >= lowwerutil:
                bidInfos.append(bidInfo)
            if bidInfo.getutil() < lowwerutil:
                break
        if len(bidInfos) == 0:
            return None
        else:
            return bidInfos[random.randint(0, len(bidInfos)-1)].getBid()

    # set<BidInfo>
    def generateRandomBids(numberOfBids, prefer, issue_num, issue_value):
        randombids = set()
        for i in range(numberOfBids):
            b = [random.choice(list(issue_value[j].keys())) for j in range(issue_num)]
            randombids.add(BidInfo(b, get_utility(b, prefer, 1, 'DISCRETE', issue_value)))
        return randombids

    def getNumberOfPosibleBids(bidSpace):
        return bidSpace
    