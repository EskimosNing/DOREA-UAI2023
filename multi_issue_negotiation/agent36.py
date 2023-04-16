from agent import Agent
from utils import get_utility
from utils import get_utility_with_discount
import random
import math
import copy
import time

class OpponentPreferences:
    def __init__(self):
        self.repeatedissue = {}  # HashMap<String, HashMap<Value, Integer>>
        self.selectedValues = None  # ArrayList
        self.opponentBids = []  # ArrayList<Agent36.BidUtility>
    
    def setRepeatedissue(self, repeatedissue):
        self.repeatedissue = repeatedissue
    
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
    
    # def __init__(self, newbid):
    #     self.bid = newbid.getBid()
    #     self.utility = newbid.getUtility()
    #     self.time = newbid.getTime()

    def setBid(self, bid):
        self.bid = bid
    
    def getBid(self):
        return self.bid

    def setUtility(self, utility):
        self.utility = utility

    def getUtility(self):
        return self.utility
    
    def setTime(self, time):
        self.time = time
    
    def getTime(self):
        return self.time


class Agent36(Agent):
    def __init__(self, max_round, name="meng-wan agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.changeTargetUtility = 0.8
        self.DamonOfferTime = 0
        self.t1 = 0.0
        self.u2 = 1.0
        self.lastBid = None
        self.round = 0
        self.oppName = None  # String
        self.Imfirst = False
        self.withDiscount = None
        self.fornullAgent = False
        self.opponentAB = None  # ArrayList<BidUtility>
        self.opponentABC = None  # ArrayList<BidUtility>
        self.oppPreferences = None  # OpponentPreferences

    def reset(self):
        super().reset()
        self.changeTargetUtility = 0.8
        self.DamonOfferTime = 0
        self.t1 = 0.0
        self.u2 = 1.0
        self.lastBid = None
        self.round = 0
        self.oppName = None  # String
        self.Imfirst = False
        self.withDiscount = False
        self.fornullAgent = False
        self.opponentAB = []  # ArrayList<BidUtility>
        self.opponentABC = []  # ArrayList<BidUtility>
        self.oppPreferences = OpponentPreferences()  # OpponentPreferences

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def receive(self, last_action=None):
        if last_action is not None:
            newBid = last_action
            newBid_util = self.getUtility(newBid)
            self.offer_received.append(newBid)
            self.utility_received.append(newBid_util)
            opBid = BidUtility(newBid, newBid_util, time.time()*1000)
            self.addBidToList(self.oppPreferences.getOpponentBids(), opBid)
            self.calculateParamForOpponent(self.oppPreferences, newBid)
            self.lastBid = newBid

    def addBidToList(self, mybids, newbid):
        if mybids is None or len(mybids) == 0:
            mybids = [newbid]
        index = len(mybids)
        for i in range(len(mybids)):
            if mybids[i].getUtility() <= newbid.getUtility():
                if mybids[i].getBid() != newbid.getBid():
                    index = i
                else:
                    return
        mybids[index] = newbid
        
    def calculateParamForOpponent(self, op, bid):
        for i in range(self.issue_num):
            if self.issue_name[i] in op.getRepeatedissue():
                vals = op.getRepeatedissue()[self.issue_name[i]]
                if bid[i] in vals:
                    repet = vals[bid[i]]
                    vals[bid[i]] = repet + 1
                else:
                    vals[bid[i]] = 1
            else:
                h = {}
                h[bid[i]] = 1
                op.getRepeatedissue()[self.issue_name[i]] = h

    def chooseWorstIssue(self):
        random_ = random.random() * 100
        sumWeight = 0.0
        minin = 1
        min = 1.0
        for i in range(self.issue_num, 0, -1):
            sumWeight += 1.0 / self.prefer[i-1]
            if self.prefer[i-1] < min:
                min = self.prefer[i-1]
                minin = i
            if sumWeight > random_:
                return i
        return minin

    def getRandomValue(self, issue_index):
        return random.choice(list(self.issue_value[issue_index].keys()))

    def getMybestBid(self, sugestBid):
        newBid = copy.deepcopy(sugestBid)
        index = self.chooseWorstIssue()
        loop = True
        bidTime = time.time() * 1000
        while loop:
            if (time.time() * 1000 - bidTime) * 1000 > 3:
                break
            newBid = copy.deepcopy(sugestBid)
            newBid[index-1] = self.getRandomValue(index-1)
            if self.getUtility(newBid) > self.getMyutility():
                return newBid
        return newBid

    def getMyutility(self):
        changeU = self.changeTargetUtility
        systemU = 1 - pow(self.relative_t, 5)
        if changeU > systemU:
            return changeU
        else:
            return systemU

    
    def offerMyNewBid(self, strategy):
        bidNN = None
        if self.opponentAB is not None and len(self.opponentAB) != 0:
            bidNN = self.getNNBid(self.opponentAB)
        if bidNN is None or self.getUtility(bidNN) < self.getMyutility():
            return self.getMybestBid(self.bestBid)
        return bidNN
    
    def gen_offer(self):
        if self.lastBid is None:
            self.Imfirst = True
            b =  self.getMybestBid(self.bestBid)
            utility_b = self.getUtility(b)
            self.offer_proposed.append(b)
            self.utility_proposed.append(utility_b)
            return b
        if self.relative_t >= 0.95:
            self.changeTargetUtility = 0.75
        if self.relative_t > 0.98:
            self.changeTargetUtility = 0.7
        
        if self.getUtility(self.lastBid) >= self.getMyutility():
            if self.relative_t >= 0.9:
                self.accept = True
                return None
        if self.relative_t >= 0.9:
            if self.opponentABC is not None and len(self.opponentABC) != 0:
                maxU = 0
                indexI = 0
                for i in range(len(self.opponentABC)):
                    if self.opponentABC[i].utility > maxU:
                        maxU = self.opponentABC[i].utility
                        indexI = i
                if self.opponentABC[indexI].utility >= self.getMyutility():
                    b = self.opponentABC[indexI].bid
                    self.opponentABC.remove(self.opponentABC[indexI])
                    self.DamonOfferTime += 1
                else:
                    b = self.offerMyNewBid(False)
                utility_b = self.getUtility(b)
                self.offer_proposed.append(b)
                self.utility_proposed.append(utility_b)
                return b
        b = self.offerMyNewBid(False)
        utility_b = self.getUtility(b)
        self.offer_proposed.append(b)
        self.utility_proposed.append(utility_b)
        return b
    
    def chooseBestIssue(self):
        random_ = random.random()
        sumWeight = 0.0
        for i in range(self.issue_num, 0, -1):
            sumWeight += self.prefer[i-1]
            if sumWeight > random_:
                return i
        return 0

    def getNNBid(self, oppAB):
        maxBid = None
        maxutility = 0.0
        size = 0
        exloop = 0
        while exloop < self.issue_sum:
            bi = self.chooseBestIssue()
            size = 0
            while oppAB is not None and len(oppAB) > size:
                b = oppAB[size].getBid()
                # newBid = copy.deepcopy(b)
                b[bi-1] = self.getRandomValue(bi-1)
                newBid = copy.deepcopy(b)
                if self.getUtility(newBid) > self.getMyutility() and self.getUtility(newBid) > maxutility:
                    maxBid = copy.deepcopy(newBid)
                    maxutility = self.getUtility(maxBid)
                size += 1
            exloop += 1
        return maxBid

    def getMutualIssues(self, strategy):
        mutualList = []
        onlyFirstFrequency = 2
        while onlyFirstFrequency > 0:
            mutualList = []
            count = 0
            for i in range(self.issue_num):
                if strategy:
                    if self.updateMutualList(mutualList, i, onlyFirstFrequency):
                        count += 1
                    if count * 2 >= self.issue_num:
                        break
                if (len(self.oppPreferences.getRepeatedissue()) == 0) or (self.issue_name[i] in self.oppPreferences.getRepeatedissue() and len(self.oppPreferences.getRepeatedissue()[self.issue_name[i]]) == 0):
                    return None
            if len(self.opponentAB) == 0:
                nullval = 0.0
                for i in range(len(mutualList)):
                    if mutualList[i] is not None:
                        nullval += 1.0
                nullval /= len(mutualList)
                if nullval >= 0.5:
                    break
            onlyFirstFrequency -= 1
        return mutualList

    def updateMutualList(self, mutualList, i, onlyFirstFrequency):
        
        if len(self.oppPreferences.getRepeatedissue()) != 0 and self.issue_name[i] in self.oppPreferences.getRepeatedissue():
            # HashMap<Value, Integer>
            valsA = self.oppPreferences.getRepeatedissue()[self.issue_name[i]]
            keys = list(valsA.keys())
            maxA = [0] * 2
            maxkeyA = [None] * 2  
            for j in range(len(keys)):
                temp = valsA[keys[j]]
                if temp > maxA[0]:
                    maxA[0] = temp
                    maxkeyA[0] = keys[j]
                elif temp > maxA[1]:
                    maxA[1] = temp
                    maxkeyA[1] = keys[j]
            # valsB == null
            if len(mutualList) > i:
                mutualList[i] = None
            else:
                mutualList.append(None)
            return False
        else:
            if len(mutualList) > i:
                mutualList[i] = None
            else:
                mutualList.append(None)
            return False

    def MyBestValue(self, issueindex):
        maxutil = 0.0
        maxvalIndex = 0
        # map: HashMap<Integer, Value>
        # bestBid map
        num = 0
        if num < len(self.issue_value[issueindex]):
            u = 0.0
            temp = copy.deepcopy(self.bestBid)
            temp[issueindex] = list(self.issue_value[issueindex].keys())[num]
            u = self.getUtility(temp)
            if u > maxutil:
                maxutil = u
                maxvalIndex = num
        return maxvalIndex
