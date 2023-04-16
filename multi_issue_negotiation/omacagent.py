from itertools import count
from numpy.lib.shape_base import split
from numpy.lib.type_check import real
from agent import Agent
from utils import get_utility, get_utility_with_discount
import random
import math
import numpy as np

class TimeBidHistory:
    def __init__(self, agent):
        self.agent = agent
        self.fMyBids = []
        self.fOpponentBids = []
        self.fTimes = []
        self.curIndex = 0
        self.curLength = 0
        self.discount = self.agent.discount
        self.est_t = 0
        self.est_u = 0
        self.maxU = -1
        self.pMaxIndex = 0
        self.bestOppBid = None
        self.maxBlock = [0.0 for _ in range(100)]  
        self.newMC = []

    def addMyBid(self, pBid):
        if pBid is None or len(pBid) == 0:
            print("pBid cannot be None or empty list")
            a = 1/0
            exit(0)
        self.fMyBids.append(pBid)
    
    def getMyBidCount(self):
        return len(self.fMyBids)

    def getMyBid(self, pIndex):
        return self.fMyBids[pIndex]

    def getMyLastBid(self):
        if self.getMyBidCount() > 0:
            return self.fMyBids[-1]
        return None

    def isInsideMyBids(self, a):
        if a in self.fMyBids:
            return True
        return False

    def addOpponentBidnTime(self, oppU, pBid, time):
        undisOppU = oppU / math.pow(self.discount, time)
        nTime = time
        if pBid is None or len(pBid) == 0:
            print("pBid cannot be None or empty list")
            a = 1/0
            exit(0)
        self.fTimes.append(time)

        if undisOppU > self.maxU:
            self.maxU = undisOppU
            self.pMaxIndex = len(self.fTimes) - 1
            self.bestOppBid = pBid
            self.newMC.append(self.pMaxIndex)
        
        if nTime >= 1.0:
            nTime = 0.99999
        if self.maxBlock[math.floor(nTime * 100)] < undisOppU:
            self.maxBlock[math.floor(nTime * 100)] = undisOppU

    def getTimeBlockList(self):
        return self.maxBlock
    
    def getOpponentBidCount(self):
        return len(self.fOpponentBids)

    def getOpponentBid(self, pIndex):
        return self.fOpponentBids[pIndex]

    def getOpponentLastBid(self):
        if self.getOpponentBidCount() > 0:
            return self.fOpponentBids[-1]
        return None

    def getMyUtility(self, b):
        return get_utility(b, self.agent.prefer, self.agent.condition, self.agent.domain_type, self.agent.issue_value)

    def getOpponentUtility(self, b):
        return get_utility(b, self.agent.prefer, self.agent.condition, self.agent.domain_type, self.agent.issue_value)

    def getFeaMC(self, time):
        lenn = len(self.newMC)
        dif = 1.0
        if lenn >= 3:
            dif = self.fTimes[self.newMC[lenn-1]] - self.fTimes[self.newMC[lenn-3]]
            dif = dif / 2.0
        elif lenn >= 2:
            dif = self.fTimes[self.newMC[lenn-1]] - self.fTimes[self.newMC[lenn-2]]
        else:
            dif = 0

        return dif

    def getMCtime(self):
        if len(self.newMC) == 0:
            return 0.0
        return self.fTimes[self.newMC[-1]]


class OMAC(Agent):
    def __init__(self, max_round, name="omac agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.actionOfPartner = None
        self.MINIMUM_UTILITY = 0.59
        self.resU = 0.0
        self.discount = 1.0
        self.EU = 0.95
        self.est_t = 0
        self.est_u = 0
        self.mBidHistory = None
        self.intervals = 100
        self.timeInt = 1.0 / self.intervals
        self.maxBid = None
        self.maxBidU = 0.0
        self.cTime = 0.0
        self.tCount = 0
        self.nextUti = 0.96
        self.numberOfIssues = 0
        self.discountThreshold = 0.845
        self.lenA = 0

        self.exma = 0
        self.est_mu = 0
        self.est_mt = 0
        self.debug = False
        self.detail = False
        self.maxTime = 180.0
        self.numberOfDiscretizationSteps = 21

    def reset(self):
        super().reset()
        self.numberOfDiscretizationSteps = 21
        self.actionOfPartner = None
        self.MINIMUM_UTILITY = 0.59
        self.resU = self.reservation
        if self.MINIMUM_UTILITY < self.resU:
            self.MINIMUM_UTILITY = self.resU * 1.06
        self.numberOfIssues = self.issue_num
        if self.domain_type == "REAL":
            self.maxBid = [self.condition for _ in range(self.issue_num)]
        elif self.domain_type == "DISCRETE":
            self.maxBid = self.bestBid
        self.maxBidU = 1
        self.cTime = 0.0
        self.tCount = 0
        self.nextUti = 0.96
        self.EU = self.EU * self.maxBidU        
        self.est_t = 0
        self.est_u = 0
        self.intervals = 100
        self.timeInt = 1.0 / self.intervals
        self.discountThreshold = 0.845
        self.lenA = 0
        self.exma = 0
        self.est_mu = 0
        self.est_mt = 0
        self.debug = False
        self.detail = False
        self.maxTime = 180.0
        if self.discount == 0.0:
            self.discount = 1.0
        self.mBidHistory = TimeBidHistory(self)

    def receive(self, last_action=None):
        if last_action is not None:
            self.actionOfPartner = last_action
            utility = get_utility(last_action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_received.append(utility)
            self.offer_received.append(last_action)
    

    def gen_offer(self, offer_type="random"):
        action = None
        if self.actionOfPartner is None:
            action = self.chooseBidAction()
        else:
            self.cTime = self.relative_t
            partnerBid = self.actionOfPartner
            offeredUtilFromOpponent = self.getUtility(partnerBid)
            self.mBidHistory.addOpponentBidnTime(offeredUtilFromOpponent, partnerBid, self.cTime)

            if self.discount < self.discountThreshold:
                if self.mBidHistory.isInsideMyBids(partnerBid):
                    self.accept = True
                    return None
            elif self.cTime > 0.97:
                if self.mBidHistory.isInsideMyBids(partnerBid):
                    self.accept = True
                    return None

            action = self.chooseBidAction()
            myOfferedUtil = self.getUtility(action)

            if self.isAcceptable(offeredUtilFromOpponent, myOfferedUtil, self.cTime, partnerBid):
                self.accept = True
                return None

        if action is not None:
            self.offer_proposed.append(action)
            action_util = get_utility(action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_proposed.append(action_util)
        return action

    def isAcceptable(self, offeredUtilFromOpponent, myOfferedUtil, time, oppBid):
        if offeredUtilFromOpponent >= myOfferedUtil:
            return True
        return False

    def chooseBidAction(self):
        nextBid = None
        if self.cTime <= 0.02:
            nextBid = self.maxBid
        else:
            nextBid = self.getFinalBid()
        
        if nextBid is None:
            self.terminate = True
            return None
        
        self.mBidHistory.addMyBid(nextBid)
        return nextBid

    def getUtility(self, offer):
        return get_utility_with_discount(offer, self.prefer, self.condition, self.domain_type, self.issue_value, self.relative_t, self.discount)

    def getFinalBid(self):
        bid = None
        upper = 1.01
        lower = 0.99
        splitFactor = 3.0
        val = 0.0
        dval = 0.0
        delay = 75
        laTime = 0
        adp = 1.2

        if self.discount >= self.discountThreshold:
            if self.cTime <= delay / 100.0:
                if self.resU <= 0.3:
                    return self.maxBid
                else:
                    val = self.EU
                dval = val * math.pow(self.discount, self.cTime)
                bid = self.genRanBid(val * lower, val * upper)
                return bid
            elif self.cTime > 0.01 * (self.tCount + delay):
                self.nextUti = self.getNextUti()
                self.tCount += 1
        else:
            if self.cTime <= self.discount / splitFactor:
                if self.resU <= 0.3:
                    return self.maxBid
                else:
                    val = self.EU
                dval = val * math.pow(self.discount, self.cTime)
                bid = self.genRanBid(val * (1.0 - 0.02), val * (1.0 + 0.02))
                return bid
            elif self.cTime > 0.01 * (self.tCount + math.floor(self.discount / splitFactor * 100)):
                self.nextUti = self.getNextUti()
                self.tCount += 1

        if self.nextUti == -3.0:
            if self.resU <= 0.3:
                return self.maxBid
            else:
                val = self.EU
        elif self.nextUti == -2.0:
             val = self.est_mu + (self.cTime - self.est_mt) * (self.est_u - self.est_mu) / (self.est_t - self.est_mt)
        elif self.nextUti == -1.0:
            val = self.getOriU(self.cTime)
        
        laTime = self.mBidHistory.getMCtime() * self.maxTime
        if self.cTime * self.maxTime - laTime > 1.5 or self.cTime > 0.995:
            dval = val * math.pow(self.discount, self.cTime)
            bid = self.genRanBid(dval * lower, dval * upper)
        else:
            if val * lower * adp >= self.maxBidU:
                bid = self.maxBid
            else:
                dval = adp * val * math.pow(self.discount, self.cTime)
                bid = self.genRanBid(dval * lower, dval * upper)

        if bid is None:
            bid = self.mBidHistory.getMyLastBid()

        if self.getUtility(self.mBidHistory.bestOppBid) >= self.getUtility(bid):
            return self.mBidHistory.bestOppBid
        
        if self.cTime > 0.999 and self.getUtility(self.mBidHistory.bestOppBid) > self.MINIMUM_UTILITY * math.pow(self.discount, self.cTime):
            return self.mBidHistory.bestOppBid
        
        return bid

    def getNextUti(self):
        utiO = self.getOriU(self.cTime + self.timeInt)
        self.exma = self.getPre()

        if self.exma >= 1.0:
            return -3.0
        if self.exma > utiO:
            self.est_t = self.cTime + self.timeInt
            self.est_u = self.exma
            self.est_mu = self.getOriU(self.cTime)
            self.est_mt = self.cTime
            return -2.0
        else:
            return -1.0

    def getOriU(self, t):
        exp = 1
        maxUtil = self.maxBidU
        minUtil = 0.69
        if minUtil < self.MINIMUM_UTILITY:
            minUtil = self.MINIMUM_UTILITY * 1.05
        e1 = 0.033
        e2 = 0.04
        time = t
        tMax = self.maxBidU
        tMin = self.MINIMUM_UTILITY * 1.05

        if self.discount >= self.discountThreshold:
            exp = minUtil + (1 - math.pow(time, 1/e1)) * (maxUtil - minUtil)
        else:
            tMax = math.pow(self.discount, 0.2)
            exp = tMin + (1 - math.pow(time, 1/e2)) * (tMax - tMin)
        
        return exp

    def getPre(self):
        lenn = 3
        pmaxList = self.mBidHistory.getTimeBlockList()
        lenA = math.floor(self.cTime * 100)
        if lenA < lenn:
            return -1.0
        
        maxList = [0.0 for _ in range(lenA)] # double array
        ma = [0.0 for _ in range(lenA)]
        res = [0.0 for _ in range(lenA)]
        exma = 0.0

        for i in range(lenA):
            maxList[i] = pmaxList[i]

        for i in range(lenn - 1):
            ma[i] = 0
        
        for i in range(lenn-1, lenA):
            ma[i] = (maxList[i] + maxList[i - 1] + maxList[i - 2]) / 3.0
        
        for i in range(lenA):
            res[i] = maxList[i] - ma[i]

        exma = ma[lenA - 1] + self.avg(res) + self.std(res) * (1.0 - math.pow(maxList[lenA - 1], 4)) * (1.3 + 0.66 * math.pow(1 - self.cTime * self.cTime, 0.4))
        return exma

    def summ(self, arr):
        summ = 0.0
        lenn = len(arr)
        for i in range(lenn):
            summ += arr[i]
        return summ
    
    def avg(self, arr):
        lenn = len(arr)
        return self.summ(arr) / lenn

    def std(self, arr):
        std = 0.0
        lenn = len(arr)
        ssum = 0.0
        for i in range(lenn):
            ssum += arr[i] * arr[i]
        std = (lenn / (lenn - 1.0)) * (ssum / lenn - math.pow(self.avg(arr), 2))
        return math.sqrt(std)


    def genRanBid(self, min, max):
        values = []  
        counter = 0
        limit = 1000
        fmax = max
        
        if self.domain_type == "REAL":
            while True:
                values = []
                for i in range(self.issue_num):
                    optionInd = random.randint(0, self.numberOfDiscretizationSteps-2)
                    values.append(0 + 1 * optionInd / self.numberOfDiscretizationSteps)
                counter += 1
                if counter > limit:
                    limit = limit + 500
                    fmax += 0.005
                
                if counter > 4000:
                    return self.mBidHistory.getMyLastBid()
                
                if self.getUtility(values) >= min and self.getUtility(values) <= fmax:
                    break            
            return values
        elif self.domain_type == "DISCRETE":
            while True:
                values = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                counter += 1
                if counter > limit:
                    limit = limit + 500
                    fmax += 0.005
                
                if counter > 4000:
                    return self.mBidHistory.getMyLastBid()
                
                if self.getUtility(values) >= min and self.getUtility(values) <= fmax:
                    break
            return values
            