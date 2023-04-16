from agent import Agent
from utils import get_utility
import random
import numpy
import math
import copy

class Atlas3(Agent):
    def __init__(self, max_round, name="atlas3 agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.negotiatingInfo = None
        self.bidSearch = None
        self.strategy = None
        self.rv = 0.0
        self.offeredBid = None
        self.supporter_num = 0
        self.CList_index = 0

    def reset(self):
        super().reset()
        self.negotiatingInfo = negotiatingInfo(self.issue_num, self.issue_value, self.prefer)
        self.bidSearch = bidSearch(self.prefer, self.issue_value, self.issue_num, self.reservation, self.negotiatingInfo)
        self.strategy = strategy(self.prefer, self.issue_num, self.issue_value, self.discount, self.reservation, self.negotiatingInfo)
        self.rv = self.reservation
        self.offeredBid = None
        self.supporter_num = 0
        self.CList_index = 0

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def receive(self, last_action=None):
        if last_action is not None:
            if self.negotiatingInfo.alreadyInitOpponent == False:
                self.negotiatingInfo.negotiatorNum = 2
                self.negotiatingInfo.initOpponent()

            utility = self.getUtility(last_action)
            self.offer_received.append(last_action)
            self.utility_received.append(utility)
            self.supporter_num = 1
            self.offeredBid = last_action
            self.negotiatingInfo.updateInfo(self.offeredBid)

            if self.supporter_num == self.negotiatingInfo.getNegotiatorNum() - 1:
                if self.offeredBid is not None:
                    self.negotiatingInfo.updatePBList(self.offeredBid)

    def gen_offer(self):
        time = math.ceil(self.t/2) / (self.max_round/2)
        self.negotiatingInfo.updateTimeScale(time)
        CList = self.negotiatingInfo.getPBList()
        if time > 1.0 - self.negotiatingInfo.getTimeScale() * (len(CList) + 1):
            return self.chooseFinalAction(self.offeredBid, CList)

        # Accept
        if self.t != 1 and self.strategy.selectAccept(self.offeredBid, time):
            self.accept = True
            return None

        # EndNegotiation
        if self.strategy.selectEndNegotiation(time):
            self.terminate = True
            return None
        
        # Offer
        return self.offerAction()

    def chooseFinalAction(self, offeredBid, CList):
        offeredBid_util = 0
        if offeredBid is not None:
            offeredBid_util = self.getUtility(offeredBid)
        if self.CList_index >= len(CList):
            if offeredBid_util >= self.rv:
                self.accept = True
                return None
            else:
                return self.offerAction()
        
        CBid = CList[self.CList_index]
        CBid_util = self.getUtility(CBid)
        if CBid_util > offeredBid_util and CBid_util > self.rv:
            self.CList_index = self.CList_index + 1
            self.negotiatingInfo.updateMyBidHistory(CBid)
            self.offer_proposed.append(CBid)
            self.utility_proposed.append(CBid_util)
            return CBid
        elif offeredBid_util > self.rv:
            self.accept = True
            return None
        
        return self.offerAction()

    def offerAction(self):
        time = math.ceil(self.t/2) / (self.max_round/2)
        randomBid = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        offerBid = self.bidSearch.getBid(randomBid, self.strategy.getThreshold(time))
        self.negotiatingInfo.updateMyBidHistory(offerBid)
        self.offer_proposed.append(offerBid)
        self.utility_proposed.append(self.getUtility(offerBid))
        return offerBid



class negotiatingInfo:
    def __init__(self, issue_num, issue_value, prefer):
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = prefer
        self.MyBidHistory = []
        self.BOBHistory = []  
        self.PBList = []  
        self.opponentsBidHistory = []
        self.opponentsAverage = 0.0  
        self.opponentsVariance = 0.0  
        self.opponentsSum = 0.0  
        self.opponentsPowSum = 0.0  
        self.opponentsStandardDeviation = 0.0  
        self.valueRelativeUtility = []  # map<Value, Double>
        self.allValueFrequency = []  # map<Value, Integer>
        self.opponentsValueFrequency = []  # map<Value, Integer>
        self.BOU = 0.0  # BestOfferedUtility
        self.MPBU = 0.0  # MaxPopularBidUtility
        self.time_scale = 0.0
        self.round = 0
        self.negotiatorNum = 2
        self.isLinerUtilitySpace = True
        self.alreadyInitOpponent = False

        self.initAllValueFrequency()
        self.initValueRelativeUtility()


    def initAllValueFrequency(self):
        for i in range(self.issue_num):
            _dict = {}
            for value in list(self.issue_value[i].keys()):
                _dict[value] = 0
            self.allValueFrequency.append(_dict)

    def initValueRelativeUtility(self):
        for i in range(self.issue_num):
            _dict = {}
            for value in list(self.issue_value[i].keys()):
                _dict[value] = 0.0
            self.valueRelativeUtility.append(_dict)

    def initOpponent(self):    
        self.alreadyInitOpponent = True
        self.initOpponentsValueFrequency()

    def initOpponentsValueFrequency(self):
        for i in range(self.issue_num):
            _dict = {}
            for value in list(self.issue_value[i].keys()):
                _dict[value] = 0
            self.opponentsValueFrequency.append(_dict)

    def updateInfo(self, offeredBid):
        self.updateNegotiatingInfo(offeredBid)
        self.updateFrequencyList(offeredBid)

    def updateNegotiatingInfo(self, offeredBid):
        self.opponentsBidHistory.append(offeredBid)
        util = get_utility(offeredBid, self.prefer, 1, 'DISCRETE', self.issue_value)
        self.opponentsSum = self.opponentsSum + util
        self.opponentsPowSum = self.opponentsPowSum + util * util
        round_num = len(self.opponentsBidHistory)
        self.opponentsAverage = self.opponentsSum / round_num
        self.opponentsVariance = self.opponentsPowSum / round_num - self.opponentsAverage * self.opponentsAverage
        if self.opponentsVariance < 0:
            self.opponentsVariance = 0.0
        self.opponentsStandardDeviation = math.sqrt(self.opponentsVariance)
        if util > self.BOU:
            self.BOBHistory.append(offeredBid)
            self.BOU = util

    def updateFrequencyList(self, offeredBid):
        for i in range(self.issue_num):
            self.opponentsValueFrequency[i][offeredBid[i]] = self.opponentsValueFrequency[i][offeredBid[i]] + 1
            self.allValueFrequency[i][offeredBid[i]] = self.allValueFrequency[i][offeredBid[i]] + 1

    def setValueRelativeUtility(self, maxBid):
        for i in range(self.issue_num):
            currentBid = copy.deepcopy(maxBid)
            for value in list(self.issue_value[i].keys()):
                currentBid[i] = value
                self.valueRelativeUtility[i][value] = get_utility(currentBid, self.prefer, 1, 'DISCRETE', self.issue_value) - get_utility(maxBid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def utilitySpaceTypeisNonLiner(self):
        self.isLinerUtilitySpace = False

    def updateMyBidHistory(self, offerBid):
         self.MyBidHistory.append(offerBid)
    
    def updateTimeScale(self, time):
        self.round = self.round + 1
        self.time_scale = time / self.round

    def takeUtility(self, elem):
        return get_utility(elem, self.prefer, 1, 'DISCRETE', self.issue_value)

    
    def updatePBList(self, popularBid):
        if popularBid not in self.PBList:
            self.PBList.append(popularBid)
            self.MPBU = max(self.MPBU, get_utility(popularBid, self.prefer, 1, 'DISCRETE', self.issue_value))
            self.PBList.sort(key=self.takeUtility, reverse=True)
            
    def getAverage(self):
        return self.opponentsAverage
    
    def getVariance(self):
        return self.opponentsVariance

    def getStandardDeviation(self):
        return self.opponentsStandardDeviation
    
    def getPartnerBidNum(self):
        return len(self.opponentsBidHistory)
    
    def getRound(self):
        return self.round

    def getNegotiatorNum(self):
        return self.negotiatorNum

    def getValueRelativeUtility(self):
        return self.valueRelativeUtility
    
    def _isLinerUtilitySpace(self):
        return self.isLinerUtilitySpace
    
    def utilitySpaceTypeisNonLiner(self):
        self.isLinerUtilitySpace = False
    
    def getBOU(self):
        return self.BOU

    def getMPBU(self):
        return self.MPBU

    def getBOBHistory(self):
        return self.BOBHistory

    def getPBList(self):
        return self.PBList

    def getTimeScale(self):
        return self.time_scale

    # Value
    def getValuebyFrequencyList(self, issue):
        current_f = 0
        max_f = 0
        max_value = None
        randomOrderValues = list(self.issue_value[issue].keys())
        numpy.random.shuffle(randomOrderValues)

        for value in randomOrderValues:
            current_f = self.opponentsValueFrequency[issue][value]
            if max_value is None or current_f > max_f:
                max_f = current_f
                max_value = value
        
        return max_value

    def getValuebyAllFrequencyList(self, issue):
        current_f = 0
        max_f = 0
        max_value = None
        randomOrderValues = list(self.issue_value[issue].keys())
        numpy.random.shuffle(randomOrderValues)

        for value in randomOrderValues:
            current_f = self.allValueFrequency[issue][value]
            if max_value is None or current_f > max_f:
                max_f = current_f
                max_value = value
        
        return max_value


class bidSearch:
    def __init__(self, prefer, issue_value, issue_num, reservation, negotiatingInfo):
        self.prefer = prefer
        self.issue_value = issue_value
        self.issue_num = issue_num
        self.reservation = reservation
        # self.domain_type = domain_type
        self.negotiatingInfo = negotiatingInfo
        self.maxBid = None  # bid

        
        self.NEAR_ITERATION = 1
        self.SA_ITERATION = 1
        self.START_TEMPERATURE = 1.0  
        self.END_TEMPERATURE = 0.0001  
        self.COOL = 0.999  
        self.STEP = 1
        self.STEP_NUM = 1

        self.initMaxBid()
        negotiatingInfo.setValueRelativeUtility(self.maxBid)

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def initMaxBid(self):
        tryNum = self.issue_num
        self.maxBid = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        for i in range(tryNum):
            self.SimulatedAnnealingSearch(self.maxBid, 1.0)
            while(self.getUtility(self.maxBid) < self.reservation):
                self.SimulatedAnnealingSearch(self.maxBid, 1.0)
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
        bid = self.getBidbyNeighborhoodSearch(baseBid, threshold) 
        if self.getUtility(bid) < threshold:
            bid = self.getBidbyAppropriateSearch(baseBid, threshold)
        if self.getUtility(bid) < threshold:          
            bid = copy.deepcopy(self.maxBid)
        bid = self.getConvertBidbyFrequencyList(bid)
        return bid
    
    
    def getBidbyNeighborhoodSearch(self, baseBid, threshold):
        bid = copy.deepcopy(baseBid)
        for i in range(self.NEAR_ITERATION):
            bid = self.NeighborhoodSearch(bid, threshold)
        return bid

    
    def NeighborhoodSearch(self, baseBid, threshold):
        currentBid = copy.deepcopy(baseBid)
        currenBidUtil = self.getUtility(baseBid)
        targetBids = []
        targetBidUtil = 0.0
        values = None

        for issueIndex in range(self.issue_num):
            values = list(self.issue_value[issueIndex].keys())
            for value in values:
                currentBid[issueIndex] = value
                currenBidUtil = self.getUtility(currentBid)
                
                if self.maxBid is None or currenBidUtil >= self.getUtility(self.maxBid):
                    self.maxBid = copy.deepcopy(currentBid)
                
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
            currentBid = copy.deepcopy(baseBid)
        
        if len(targetBids) == 0:
            return baseBid
        else:
            return targetBids[random.randint(0, len(targetBids)-1)]


    def getBidbyAppropriateSearch(self, baseBid, threshold):
        bid = copy.deepcopy(baseBid)

        if self.negotiatingInfo._isLinerUtilitySpace():
            bid = self.relativeUtilitySearch(threshold)
            if self.getUtility(bid) < threshold:
                self.negotiatingInfo.utilitySpaceTypeisNonLiner()
        
        if self.negotiatingInfo._isLinerUtilitySpace() == False:
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

    
    def getConvertBidbyFrequencyList(self, baseBid):
        currentBid = copy.deepcopy(baseBid)
        randomOrderIssues = []
        for i in range(self.issue_num):
            randomOrderIssues.append(i)
        numpy.random.shuffle(randomOrderIssues)
        for issueIndex in randomOrderIssues:
            nextBid = copy.deepcopy(currentBid)
            nextBid[issueIndex] = self.negotiatingInfo.getValuebyAllFrequencyList(issueIndex)
            if self.getUtility(nextBid) >= self.getUtility(currentBid):
                currentBid = copy.deepcopy(nextBid)
        return currentBid

    def relativeUtilitySearch(self, threshold):
        bid = copy.deepcopy(self.maxBid)
        d = threshold - 1.0
        concessionSum = 0.0
        relativeUtility = 0.0
        valueRelativeUtility = self.negotiatingInfo.getValueRelativeUtility()
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


    def getRandomBid(self, threshold):
        pass

    #List<Issue>
    def criticalIssue(self, baseBid):
        currentBid = copy.deepcopy(baseBid)
        criticalIssues = []
        values = None
        
        for issueIndex in range(self.issue_num):
            values = list(self.issue_value[issueIndex].keys())
            for value in values:
                currentBid[issueIndex] = value
                if self.getUtility(currentBid) != self.getUtility(baseBid):
                    criticalIssues.append(issueIndex)
                    break
            currentBid = copy.deepcopy(baseBid)
        
        return criticalIssues


class strategy:
    def __init__(self, prefer, issue_num, issue_value, discount, reservation, negotiatingInfo):
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = prefer
        self.negotiatingInfo = negotiatingInfo
        self.df = discount
        self.rv = reservation
        self.A11 = 0.0
        self.A12 = 0.0
        self.A21 = 0.0
        self.A22 = 0.0
        self.TF = 1.0
        self.PF = 0.5

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def selectAccept(self, offeredBid, time):
        offeredBidUtil = self.getUtility(offeredBid)
        # print('offeredBidUtil:', offeredBidUtil, 'self.getThreshold(time):', self.getThreshold(time), 'time:', time)
        if offeredBidUtil >= self.getThreshold(time):
            return True
        else:
            return False

    def getThreshold(self, time):
        threshold = 1.0
        self.updateGameMatrix()
        target = self.getExpectedUtilityinFOP() / math.pow(self.df, time)
        if self.df == 1.0:
            threshold = target + (1.0 - target) * (1.0 - time)
        else:
            threshold = max(1.0 - time / self.df, target)
        return threshold

    def selectEndNegotiation(self, time):
        if self.rv * math.pow(self.df, time) >= self.getThreshold(time):
            return True
        else:
            return False

    def updateGameMatrix(self):
        if self.negotiatingInfo.getNegotiatorNum() == 2:
            C = self.negotiatingInfo.getBOU()
        else:
            C = self.negotiatingInfo.getMPBU()
            print('negotiator num is wrong.')
            exit(-1)
        self.A11 = self.rv * self.df
        self.A12 = math.pow(self.df, self.TF)
        if C >= self.rv:
            self.A21 = C * math.pow(self.df, self.TF)
        else:
            self.A21 = self.rv * self.df
        self.A22 = self.PF * self.A21 + (1.0-self.PF) * self.A12


    def getExpectedUtilityinFOP(self):
        q = self.getOpponentEES()
        return q * self.A21 + (1-q) * self.A22

    def getOpponentEES(self):
        q = 1.0
        if (self.A12 - self.A22 != 0) and (1.0 - (self.A11 - self.A21) / (self.A12 - self.A22) != 0):
            q = 1.0 / (1.0 - (self.A11 - self.A21) / (self.A12 - self.A22))
        if q < 0.0 or q > 1.0:
            q = 1.0
        return q
