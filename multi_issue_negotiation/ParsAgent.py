from numpy.lib import twodim_base
from numpy.lib.function_base import append
from agent import Agent
from utils import get_utility
from utils import get_utility_with_discount
import random
import math
import copy
import time

class OpponentPreferences:
    def __init__(self, issue_value):
        self.repeatedissue = copy.deepcopy(issue_value)  # HashMap<Value, Integer>
        for i in range(len(issue_value)):
            for key in self.repeatedissue[i].keys():
                self.repeatedissue[i][key] = 0
        self.selectedValues = []
        self.opponentBids = []

    def getRepeatedissue(self):
        return self.repeatedissue

    def setSelectedValues(self, selectedValues):
        self.selectedValues = selectedValues

    def getSelectedValues(self):
        return self.selectedValues

    def setOpponentBids(self, opponentBids):
        self.opponentBids = opponentBids

    def getOpponentBids(self):
        return self.opponentBids


class BidUtility:
    def __init__(self, b, u, t):
        self.bid = b
        self.utility = u
        self.time = t
        
    def setBid(self, bid):
        self.bid = bid
    
    def getBid(self):
        return self.bid

    def setUtility(self, u):
        self.utility = u

    def getUtility(self):
        return self.utility

    def setTime(self, time):
        self.time = time
    
    def getTime(self):
        return self.time


class ParsAgent(Agent):
    def __init__(self, max_round, name="ParsAgent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.lastBid = None
        self.round = 0
        self.myutility = 0.8
        self.Imfirst = False
        self.withDiscount = None
        # self.fornullAgent = False
        self.opponentAB = []  # BidUtility
        self.oppAPreferences = None

    def reset(self):
        super().reset()
        self.lastBid = None
        self.round = 0
        self.myutility = 0.8
        self.Imfirst = False
        if self.discount == 1.0:
            self.withDiscount = False
        else:
            self.withDiscount = True
        # self.fornullAgent = False
        self.opponentAB = []
        self.oppAPreferences = OpponentPreferences(self.issue_value)

    def receive(self, last_action):
        if last_action is not None:
            newBid = last_action
            utililty = self.getUtility(newBid)
            self.offer_received.append(newBid)
            self.utility_received.append(utililty)
            opBid = BidUtility(newBid, self.getUtility(newBid), self.t)
            self.addBidToList(self.oppAPreferences.getOpponentBids(), opBid)
            self.calculateParamForOpponent(self.oppAPreferences, newBid)
            self.lastBid = newBid

    def getRandomValue(self, issue_index):
        return random.choice(list(self.issue_value[issue_index].keys()))

    def current_milli_time(self):
        return round(time.time() * 1000)

    def chooseWorstIssue(self):
        ran = random.random() * 100
        sumWeight = 0
        minin = 0
        minn = 1.0
        i = self.issue_num - 1
        while(i >= 0):
            sumWeight += 1/self.issue_weight[i]
            if self.issue_weight[i] < minn:
                minn = self.issue_weight[i]
                minin = i
            if sumWeight > ran:
                return i
            i -= 1
        return minin
        

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)

    def getUtilityWithDiscount(self, bid, time):
        return get_utility_with_discount(bid, self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount)

    def addBidToList(self, mybids, newbid):
        index = len(mybids)
        for i in range(index):
            if mybids[i].getUtility() <= newbid.getUtility():
                if mybids[i].getBid() != newbid.getBid():
                    index = i
                else:
                    return
        mybids.insert(index, newbid)

    def calculateParamForOpponent(self, op, bid):
        for i in range(self.issue_num):
            if len(op.getRepeatedissue()) <= i:
                # create a dict
                h = {}
                h[bid[i]] = 1
                op.append(copy.deepcopy(h))
            else:
                if bid[i] in op.getRepeatedissue()[i]:
                    op.getRepeatedissue()[i][bid[i]] += 1 
                else:
                    op.getRepeatedissue()[i][bid[i]] = 1

    def getE(self):
        if self.withDiscount:
            return 0.2
        return 0.15

    def f(self, t):
        if self.getE() == 0:
            return 0
        return math.pow(t, 1/self.getE())

    def getTargetUtility(self):
        offset = 1 / self.max_round
        target = 1 - self.f(self.relative_t - offset)
        return target
    
    def getMyutility(self):
        myutility = self.getTargetUtility()
        if myutility < 0.7:
            return 0.7
        return myutility

    def getMybestBid(self, sugestBid, time):
        newBid = copy.deepcopy(sugestBid)
        index = self.chooseWorstIssue()
        loop = True
        bidTime = self.current_milli_time()
        while loop:
            if (self.current_milli_time() - bidTime) * 1000 > 3:
                break
            newBid = copy.deepcopy(sugestBid)
            newBid[index] = self.getRandomValue(index)
            if self.getUtility(newBid) > self.getMyutility():
                return newBid
        return newBid

    def chooseBestIssue(self):
        ran = random.random()
        sumWeight = 0
        i = self.issue_num - 1
        while i >= 0:
            sumWeight += self.issue_weight[i]
            if sumWeight > ran:
                return i
            i -= 1
        return 0

    def getNNBid(self, oppAB):
        maxBid = None
        maxutility = 0
        size = 0
        exloop = 0
        while exloop < self.issue_num:
            bi = self.chooseBestIssue()
            size = 0
            while oppAB is not None and len(oppAB) > size:
                b = oppAB[size].getBid()
                newBid = copy.deepcopy(b)
                newBid[bi] = self.getRandomValue(bi)
                if self.getUtility(newBid) > self.getMyutility() and self.getUtility(newBid) > maxutility:
                    maxBid = copy.deepcopy(newBid)
                    maxutility = self.getUtility(maxBid)
                size += 1
            exloop += 1
        return maxBid



    def updateMutualList(self, mutualList, twocycle, i):
        # if len(self.oppAPreferences.getRepeatedissue()) > i:
        vals = self.oppAPreferences.getRepeatedissue()[i]
        keys = list(vals.keys())
        maxCount = [0, 0]
        maxkey = [None, None]
        for j in range(len(keys)):
            temp = vals[keys[j]]  # The number of occurrences of the jth value of the ith issue.
            if temp > maxCount[0]:
                maxCount[0] = temp
                maxkey[0] = keys[j]
            elif temp > maxCount[1]:
                maxCount[1] = temp
                maxkey[1] = keys[j]
        

    def getMutualIssues(self):
        mutualList = []  
        twocycle = 2
        while twocycle > 0:
            mutualList = []
            for i in range(self.issue_num):
                self.updateMutualList(mutualList, twocycle, i)


    def MyBestValue(self, issueindex): 
        maxutil = 0
        maxvalIndex = 0
        map = self.bestBid
        num = 0

        for key in self.issue_value[issueindex].keys(): # 60GB,90GB,120GB
            temp = copy.deepcopy(map)
            temp[issueindex] = key
            u = self.getUtility(temp)
            if u > maxutil:
                maxutil = u
                maxvalIndex = num
            break
            num += 1

        return maxvalIndex

    def offerMyNewBid(self):
        bidNN = None
        if self.opponentAB is not None and len(self.opponentAB) != 0:
            bidNN = self.getNNBid(self.opponentAB)
        
        if bidNN is None or self.getUtility(bidNN) < self.getMyutility():
            bid = []
            for i in range(self.issue_num):
                bid.append(list(self.issue_value[i].keys())[self.MyBestValue(i)])
            if self.getUtility(bid) > self.getMyutility():
                return bid
            else:
                return self.getMybestBid(self.bestBid, 0)
        else:
            return bidNN
        
        return None

    def gen_offer(self):
        if self.lastBid is None or len(self.lastBid) == 0:
            self.Imfirst = True
            b = self.getMybestBid(self.bestBid, 0)
            self.offer_proposed.append(b)
            self.utility_proposed.append(self.getUtility(b))
            return b
        else:
            if self.getUtility(self.lastBid) > self.getMyutility():
                self.accept = True
                return None
            else:
                b = self.offerMyNewBid()
                if self.getUtility(b) < self.getMyutility():
                    offer = self.getMybestBid(self.bestBid, 0)
                    self.offer_proposed.append(offer)
                    self.utility_proposed.append(self.getUtility(offer))
                    return offer
                else:
                    self.offer_proposed.append(b)
                    self.utility_proposed.append(self.getUtility(b))
                    return b
