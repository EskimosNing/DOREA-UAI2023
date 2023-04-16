from agent import Agent
from utils import get_utility
import random
import math
import copy
import numpy

class MyAgent(Agent):
    def __init__(self, max_round, name="MyAgent agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.negotiationInfo = None
        self.bidSearch = None
        self.negotiationStrategy = None
        self.offeredBid = None

    def reset(self):        
        super().reset()
        self.negotiationInfo = negotiationInfo(self.issue_num, self.issue_value, self.prefer)
        self.bidSearch = bidSearch(self.reservation, self.issue_num, self.issue_value, self.prefer, self.negotiationInfo)
        self.negotiationStrategy = negotiationStrategy(self.discount, self.reservation, self.issue_num, self.issue_value, self.prefer, self.negotiationInfo, self.max_round)
        self.offeredBid = None

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)

    def getRandomBid(self):
        offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        return offer

    def receive(self, last_action):
        if last_action is not None:
            last_util = self.getUtility(last_action)
            self.offer_received.append(last_action)
            self.utility_received.append(last_util)
            self.negotiationInfo.updateOpponentsNum(2)
            self.offeredBid = last_action
            self.negotiationInfo.updateInfo(self.offeredBid)
            self.negotiationInfo.updateOpponentIssueWeight(self.offeredBid)
            
    def gen_offer(self):
        time = self.relative_t
        if self.offer_received is not None and len(self.offer_received) > 0 and self.negotiationStrategy.selectAccept(self.offeredBid, time):
            self.accept = True
            return None
        self.negotiationInfo.updateLast(False)
        if self.offer_received is not None and len(self.offer_received) > 0 and self.negotiationStrategy.selectEndNegotiation(time):
            self.terminate = True
            return None
        offerBid = self.bidSearch.getBid(self.getRandomBid(), self.negotiationStrategy.getThreshold(self.relative_t))
        self.offeredBid = offerBid
        self.negotiationInfo.updateMyBidHistory(offerBid)
        offerBid_util = self.getUtility(offerBid)
        self.offer_proposed.append(offerBid)
        self.utility_proposed.append(offerBid_util)
        return offerBid

class negotiationInfo:
    def __init__(self, issue_num, issue_value, perfer):
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = perfer
        # self.opponents = None  
        self.MyBidHistory = [] # list
        self.opponentBidHistory = [] # list
        self.opponentAverage = 0.0 # double
        self.opponentVariance = 0.0 # double
        self.opponentSum = 0.0 # double
        self.opponentPowSum = 0.0 # double
        self.opponentStandardDeviation = 0.0 # double
        self.valueRelativeUtility = [] # list<map-Value,Double> => self.issue_value
        self.round = 0
        self.negotiatorNum = 0
        self.isLinerUtilitySpace = True
        self.isLast = False
        self.opponentAccept = [] # list<Bid>
        self.opponentIssueWeight = [] # list<map-Value,Int>
        self.insurance = None # Bid
        self.initValueRelativeUtility()

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def initValueRelativeUtility(self):
        for i in range(self.issue_num):
            _dict = {}
            for value in list(self.issue_value[i].keys()):
                _dict[value] = 0.0
            self.valueRelativeUtility.append(_dict)

    def updateInfo(self, offeredBid):
        self.updateNegotiatingInfo(offeredBid)

    def updateNegotiatingInfo(self, offeredBid):
        self.opponentBidHistory.append(offeredBid)
        util = self.getUtility(offeredBid)
        self.opponentSum += util
        self.opponentPowSum += pow(util, 2)
        round_num = len(self.opponentBidHistory)
        self.opponentAverage = self.opponentSum / round_num
        self.opponentVariance = self.opponentPowSum/round_num - pow(self.opponentAverage, 2)
        if self.opponentVariance < 0:
            self.opponentVariance = 0.0
        self.opponentStandardDeviation = math.sqrt(self.opponentVariance)

    def setValueRelativeUtility(self, maxBid):
        for i in range(self.issue_num):
            currentBid = copy.deepcopy(maxBid)
            for value in list(self.issue_value[i].keys()):
                currentBid[i] = value
                self.valueRelativeUtility[i][value] = get_utility(currentBid, self.prefer, 1, 'DISCRETE', self.issue_value) - get_utility(maxBid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def updateOpponentsNum(self, num):
        self.negotiatorNum = num
    
    def utilitySpaceTypeisNonLiner(self):
        self.isLinerUtilitySpace = False
    
    def updateMyBidHistory(self, offerBid):
        self.MyBidHistory.append(offerBid)

    def getAverage(self):
        return self.opponentAverage
    
    def getVariance(self):
        return self.opponentVariance

    def getStandardDeviation(self):
        return self.opponentStandardDeviation
    
    def getPartnerBidNum(self):
        return len(self.opponentBidHistory)
    
    def getRound(self):
        return self.round

    def getNegotiatorNum(self):
        return self.negotiatorNum

    def getValueRelativeUtility(self):
        return self.valueRelativeUtility
    
    def _isLinerUtilitySpace(self):
        return self.isLinerUtilitySpace

    def getValues(self, issue_index):
        return list(self.issue_value[issue_index].keys())

    def _isLast(self):
        return self.isLast

    def updateLast(self, b):
        self.isLast = b

    def updateAcceptList(self, acceptedBid):
        self.opponentAccept.append(acceptedBid)

    def getAcceptList(self):
        return self.opponentAccept

    def getAcceptedNum(self):
        return len(self.opponentAccept)

    def isCoward(self):
        return self.getAcceptedNum() > 3
    
    def isArrogant(self):
        return self.getAcceptedNum() <= 2

    def getLeastValue(self):
        least = 0.0
        if self.isArrogant():
            ave = self.getAverage()
            var = self.getVariance()
            temp = ave + var * 0.1
            if temp > least:
                least = temp
        return least
    
    def getOpponentHistory(self, num):
        _len = len(self.opponentBidHistory)
        recentHistory = []
        if num > len(self.opponentBidHistory):
            anum = len(self.opponentBidHistory)
        else:
            anum = num
        for i in range(anum):
            recentHistory.append(self.opponentBidHistory[_len-1-i])
        return recentHistory

    def getOpponentBidNum(self):
        return len(self.opponentBidHistory)

    def getWeight(self, issue_index):
        if len(self.opponentIssueWeight) == 0:
            self.opponentIssueWeight = self.initIssueWeight()
        return self.opponentIssueWeight[issue_index]

    def getWeights(self):
        return self.opponentIssueWeight

    def getOpponentMaxValues(self):
        opponentMaxValues = [] # list<map>
        for i in range(self.issue_num):
            maxValue = None  # Value
            sum = 0
            for value in list(self.getWeight(i).keys()):
                if maxValue is None:
                    maxValue = value
                elif self.getCount(i, maxValue) < self.getCount(i, value):
                    maxValue = value
                sum += self.getCount(i, value)
            
            temp = {}
            if sum == 0:
                temp[maxValue] = 0
            else:
                temp[maxValue] = self.getCount(i, maxValue) / sum
            opponentMaxValues.append(temp)
        return opponentMaxValues

    def getCount(self, issue_index, value):
        return self.getWeight(issue_index)[value]

    def initRecentWeight(self):
        # list<map>
        recentWeight = []
        for i in range(self.issue_num):
            weights = {}
            for value in self.getValues(i):
                weights[value] = 0.0
            recentWeight.append(weights)
        return recentWeight

    def getRecentWeight(self, num):
        # list<map>
        recentWeight = self.initRecentWeight()
        recentCount = self.getRecentCount(num)
        for i in range(self.issue_num):
            sum = 0
            for value in self.getValues(i):
                sum += recentCount[i][value]
            for value in self.getValues(i):
                if sum == 0:
                    recentWeight[i][value] = 0.0
                else:
                    recentWeight[i][value] = (recentCount[i][value] + 0.0) / sum
        return recentWeight

    def getRecentCount(self, num):
        recentCount = self.initIssueWeight()
        for bid in self.getOpponentHistory(num):
            for i in range(self.issue_num):
                v = bid[i]
                c = recentCount[i][v]
                recentCount[i][v] = c + 1
        return recentCount

    def initIssueWeight(self):
        # list<map>
        IssueWeight = []
        for i in range(self.issue_num):
            values = {}
            for value in self.getValues(i):
                values[value] = 0
            IssueWeight.append(values)
        return IssueWeight

    def getRecentMaxWeight(self, num):
        # list<map>
        recentMaxWeight = self.initRecentWeight()
        recentWeight = self.getRecentWeight(num)
        for i in range(self.issue_num):
            maxValue = None
            for key, value in recentWeight[i].items():
                if maxValue is None:
                    maxValue = key
                else:
                    if value > recentWeight[i][maxValue]:
                        maxValue = key
            weight = {}
            weight[maxValue] = recentWeight[i][maxValue]
            recentMaxWeight.append(weight)
        return recentMaxWeight
    
    def getSlant(self):
        num = 20
        # list<map>
        allWeight = self.getOpponentMaxValues()
        recentWeight = self.getRecentMaxWeight(num)
        sumSlant = 0.0
        for i in range(self.issue_num):
            for v in list(allWeight[i].keys()):
                if v not in recentWeight[i]:
                    continue
                rawSlant = allWeight[i][v] - recentWeight[i][v]
                if rawSlant < 0:
                    continue
                else:
                    slant = rawSlant * allWeight[i][v]
                    sumSlant += slant
        return sumSlant

    def updateOpponentIssueWeight(self, bid):
        if len(self.opponentIssueWeight) == 0:
            self.opponentIssueWeight = self.initIssueWeight()
        for i in range(self.issue_num):
            current = self.opponentIssueWeight[i][bid[i]]
            self.opponentIssueWeight[i][bid[i]] = current + 1


class bidSearch:
    def __init__(self, reservation, issue_num, issue_value, prefer, negotiationInfo):
        
        self.NEAR_ITERATION = 1
        self.SA_ITERATION = 1
        self.START_TEMPERATURE = 1.0  
        self.END_TEMPERATURE = 0.0001  
        self.COOL = 0.999  
        self.STEP = 1
        self.STEP_NUM = 1
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = prefer
        self.reservation = reservation
        self.negotiationInfo = negotiationInfo
        self.maxBid = None
        self.initMaxBid()
        self.negotiationInfo.setValueRelativeUtility(self.maxBid)        

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def getRandomBid(self):
        offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        return offer

    def initMaxBid(self):
        tryNum = self.issue_num
        self.maxBid = self.getRandomBid()
        for i in range(tryNum):
            while True:
                self.SimulatedAnnealingSearch(self.maxBid, 1.0)
                if self.getUtility(self.maxBid) >= self.reservation:
                    break
            if self.getUtility(self.maxBid) == 1.0:
                break

    def SimulatedAnnealingSearch(self, baseBid, threshold):
        currentBid = copy.deepcopy(baseBid) 
        currenBidUtil = self.getUtility(baseBid)
        nextBid = None  
        nextBidUtil = 0.0
        targetBids = []
        targetBidUtil = 0.0
        currentTemperature = self.START_TEMPERATURE
        newCost = 1.0
        currentCost = 1.0

        while currentTemperature > self.END_TEMPERATURE:
            nextBid = copy.deepcopy(currentBid)
            for i in range(self.STEP_NUM):
                issueIndex = random.randint(0, self.issue_num-1)
                values = list(self.issue_value[issueIndex].keys())
                valueIndex = random.randint(0, len(values)-1)
                nextBid[issueIndex] = values[valueIndex]
                nextBidUtil = self.getUtility(nextBid)
                if self.maxBid is None or nextBidUtil >= self.getUtility(self.maxBid):
                    self.maxBid = copy.deepcopy(nextBid)
            
            newCost = abs(threshold - nextBidUtil)
            currentCost = abs(threshold - currenBidUtil)
            p = math.exp(-abs(newCost - currentCost) / currentTemperature)
            if newCost < currentCost or p > random.random():
                currentBid = copy.deepcopy(nextBid)
                currenBidUtil = nextBidUtil
            
            
            if currenBidUtil >= threshold:
                if len(targetBids) == 0:
                    targetBids.append(copy.deepcopy(currentBid))
                    targetBidUtil = self.getUtility(currentBid)
                else:
                    if currenBidUtil < targetBidUtil:
                        targetBids = []
                        targetBids.append(copy.deepcopy(currentBid))
                        targetBidUtil = self.getUtility(currentBid)
                    elif currenBidUtil == targetBidUtil:
                        targetBids.append(copy.deepcopy(currentBid))
            
            currentTemperature = currentTemperature * self.COOL
        
        if len(targetBids) == 0:
            return baseBid
        else:
            return targetBids[random.randint(0, len(targetBids)-1)]

    def getBid(self, baseBid, threshold):
        bid = self.getBidbyAppropriateSearch(baseBid, threshold)
        if self.getUtility(bid) < threshold:          
            bid = copy.deepcopy(self.maxBid)
        return bid

    def getBidbyAppropriateSearch(self, baseBid, threshold):
        bid = copy.deepcopy(baseBid)

        if self.negotiationInfo._isLinerUtilitySpace():
            bid = self.relativeUtilitySearch(threshold)
            if self.getUtility(bid) < threshold:
                self.negotiationInfo.utilitySpaceTypeisNonLiner()
        
        if self.negotiationInfo._isLinerUtilitySpace() == False:
            currentBid = None
            currentBidUtil = 0
            min = 1.0
            for i in range(self.SA_ITERATION):
                currentBid = self.SimulatedAnnealingSearch(bid, threshold)
                currentBidUtil = self.getUtility(currentBid)
                if currentBidUtil <= min and currentBidUtil >= threshold:
                    bid = copy.deepcopy(currentBid)
                    min = currentBidUtil
        
        return bid

    def relativeUtilitySearch(self, threshold):
        bid = copy.deepcopy(self.maxBid)
        d = threshold - 1.0
        concessionSum = 0.0
        relativeUtility = 0.0
        valueRelativeUtility = self.negotiationInfo.getValueRelativeUtility()
        randomOrderIssues = []
        for i in range(self.issue_num):
            randomOrderIssues.append(i)
        numpy.random.shuffle(randomOrderIssues)
        randomValues = None
        for issueIndex in randomOrderIssues:
            randomValues = list(self.issue_value[issueIndex].keys())
            numpy.random.shuffle(randomValues)
            for value in randomValues:
                relativeUtility = valueRelativeUtility[issueIndex][value]
                if d <= concessionSum + relativeUtility:
                    bid[issueIndex] = value
                    concessionSum = concessionSum + relativeUtility
                    break
        return bid


class negotiationStrategy:
    def __init__(self, discount, reservation, issue_num, issue_value, prefer, negotiationInfo, max_round):
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = prefer 
        self.negotiationInfo = negotiationInfo
        self.max_round = max_round
        self.df = -1.0
        self.rv = reservation
        self.discount = discount

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)
    
    def selectAccept(self, offeredBid, time):
        if self.df == -1.0:
            util = self.getUtility(offeredBid)
            discounted = util * pow(self.discount, time)
            if util == 0:
                self.df = self.discount
            else:
                self.df = pow(discounted/util, 1/time)
        if self.getUtility(offeredBid) >= self.getThreshold(time):
            return True
        else:
            return False

    def selectEndNegotiation(self, time):
        if self.getThreshold(time) < self.rv:
            return True
        return False
    
    def getThreshold(self, time):
        threshold = 0.95
        e = 0.01
        threshold = 1.0 - pow(time, 1 / e)
        arrogant = self.negotiationInfo.isArrogant()
        if arrogant and time != 1/self.max_round:
            slant = self.negotiationInfo.getSlant()
            rushValue = threshold * self.df * slant
            concedeValue = self.negotiationInfo.getAverage() * math.pow(self.df, time)
            if concedeValue > rushValue:
                if self.negotiationInfo.getOpponentBidNum() < 70:
                    e = 0.5
                elif slant < 0.15:
                    if self.df == 1.0:
                        e = 0.3
                    else:
                        e = 1.65
                else:
                    e = 0.1

        threshold = 1.0 - pow(time, 1 / e)
        return threshold
