import math
from agent import Agent
from utils import get_utility
import random

class FrequencyBasedOpponentModel:
    def __init__(self):
        self.issue_num = 0  # domainIssues
        self.issueValueFrequency = []  # List<Map<String, Integer>>
        self.totalNumberOfBids = 0
        self.issueCount = 0
        self.issueWeight = []
        self.issue_value = None
    
    def init(self, issue_num, issue_value):
        self.issue_num = issue_num
        self.issueCount = issue_num
        self.issue_value = issue_value
        self.issueWeight = []
        for i in range(self.issueCount):
            self.issueWeight.append(0)
        for i in range(self.issueCount):
            i_dict = self.issue_value[i]
            i_issue_values = list(i_dict.keys())
            numberOfValues = len(i_issue_values)
            i_value_frenquency_dict = {}
            for j in range(numberOfValues):
                i_value_frenquency_dict[i_issue_values[j]] = 0
            self.issueValueFrequency.append(i_value_frenquency_dict)

    def updateModel(self, bid, numberOfBids):
        if bid is None:
            return
        self.totalNumberOfBids = numberOfBids
        for i in range(self.issueCount):
            value = bid[i]
            currentValue = self.issueValueFrequency[i][value]
            currentValue += 1
            self.issueValueFrequency[i][value] = currentValue
            self.updateIssueWeight()
    
    def updateIssueWeight(self):
        for i in range(self.issueCount):
            i_dict = self.issueValueFrequency[i]
            self.issueWeight[i] = self.calculateStandardDeviation(i_dict)

    def calculateStandardDeviation(self, i_dict):
        sum = 0
        # standardDeviation
        size = len(i_dict)
        for key, value in i_dict.items():
            sum += value
        mean = sum / size
        sum2 = 0
        for key, value in i_dict.items():
            sum2 += pow(mean-value, 2)
        if sum2 != 0:
            standardDeviation = math.sqrt(sum2/size)
        else:
            standardDeviation = 0
        return standardDeviation

    def getUtility(self, bid):
        if self.totalNumberOfBids == 0:
            return 0
        sumOfEachIssueUtility = 0
        sumOfIssueWeight = 0
        for i in range(self.issueCount):
            sumOfIssueWeight += self.issueWeight[i]
        for i in range(self.issueCount):
            value = bid[i]
            numberOfPreOffers = self.issueValueFrequency[i][value]
            sumOfEachIssueUtility += (numberOfPreOffers/ self.totalNumberOfBids) * (self.issueWeight[i]/sumOfIssueWeight)
        return sumOfEachIssueUtility / self.issueCount

class AgreeableAgent2018(Agent):
    def __init__(self, max_round, name="agreeable agent 2018", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.timeToConcede = 0.2
        self.smallDomainUpperBound = 1000
        self.midDomainUpperBound = 10000
        self.timeForUsingModelForSmallDomain = 0.2
        self.timeForUsingModelForMidDomain = 0.3
        self.timeForUsingModelForLargeDomain = 0.4
        self.neigExplorationDisFactor = 0.05
        self.concessionFactor = 0.1
        # For k = 0 the agent starts with a bid of maximum utility
        self.k = 0
        self.minimumUtility = 0.8
        self.domainSize = self.bidSpace
        self.pMin = 0
        self.pMax = 0
        self.lastReceivedOffer = None
        self.opponentModel = None
        self.opponentBidCount = 0
        self.timeForUsingModel = 0.1
        self.canUseModel = True

    def reset(self):
        super().reset()
        self.timeToConcede = 0.2
        self.smallDomainUpperBound = 1000
        self.midDomainUpperBound = 10000
        self.timeForUsingModelForSmallDomain = 0.2
        self.timeForUsingModelForMidDomain = 0.3
        self.timeForUsingModelForLargeDomain = 0.4
        self.neigExplorationDisFactor = 0.05
        self.concessionFactor = 0.1
        # For k = 0 the agent starts with a bid of maximum utility
        self.k = 0
        self.minimumUtility = 0.8
        self.domainSize = self.bidSpace
        worstBid = []
        for i in range(self.issue_num):
            min_value = 2
            min_key = None
            i_dict = self.issue_value[i]
            for key,value in i_dict.items():
                if value < min_value:
                    min_value = value
                    min_key = key
            worstBid.append(min_key)
        self.pMin = self.getUtility(worstBid)
        self.pMax = self.getUtility(self.bestBid)
        self.lastReceivedOffer = None        
        self.opponentBidCount = 0
        self.timeForUsingModel = 0.1
        self.canUseModel = True
        self.determineTimeForUsingModel()
        self.opponentModel = FrequencyBasedOpponentModel()
        self.opponentModel.init(self.issue_num, self.issue_value)

    def determineTimeForUsingModel(self):
        if self.domainSize < self.smallDomainUpperBound:
            self.timeForUsingModel = self.timeForUsingModelForSmallDomain
        elif self.domainSize < self.midDomainUpperBound:
            self.timeForUsingModel = self.timeForUsingModelForMidDomain
        else:
            self.timeForUsingModel = self.timeForUsingModelForLargeDomain

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def receive(self, last_action=None):
        if last_action is not None:
            if self.canUseModel:
                self.opponentBidCount += 1
                self.opponentModel.updateModel(last_action, self.opponentBidCount)
            self.lastReceivedOffer = last_action
            utility = self.getUtility(last_action)
            self.offer_received.append(last_action)
            self.utility_received.append(utility)

    def gen_offer(self):
        if self.lastReceivedOffer is None:
            best_u = self.getUtility(self.bestBid)
            self.offer_proposed.append(self.bestBid)
            self.utility_proposed.append(best_u)
            return self.bestBid
        else:
            myBid = self.getNextBid()
            utility1 = self.getUtility(self.lastReceivedOffer)
            utility2 = self.getUtility(myBid)
            if self.isAcceptable(utility1, utility2):
                self.accept = True
                return None
            else:
                self.offer_proposed.append(myBid)
                self.utility_proposed.append(utility2)
                return myBid

    def isAcceptable(self, opponentUtility, myBidUtilByTime):
        if opponentUtility >= myBidUtilByTime:
            return True
        time = self.relative_t
        return time >= 0.99 and opponentUtility >= self.reservation
    
    def getUtilityByTime(self, time):
        if time < self.timeToConcede:
            return 1
        else:
            time = (time - self.timeToConcede) / (1 - self.timeToConcede)
            return self.pMin + (self.pMax - self.pMin) * (1 - self.f(time))

    def f(self, time):
        if self.concessionFactor == 0:
            return self.k
        return self.k + (1 - self.k) * pow(time, 1.0 / self.concessionFactor)

    def isModelUsable(self):
        time = self.relative_t
        return time >= self.timeForUsingModel

    def getBidNearUtility(self, targetUtility):
        res_offer = None
        distance = 100
        if self.allBids is None:
            self.allBids = self.getAllBids()
        for _bid in self.allBids:
            utility = get_utility(_bid, self.prefer, self.condition, self.domain_type, self.issue_value)
            if abs(targetUtility - utility) < distance:
                distance = abs(targetUtility - utility)
                res_offer = _bid
        return res_offer

    def getBidsinRange(self, lowerbound, upperbound):
        bidsInRange = []
        if self.allBids is None:
            self.allBids = self.getAllBids()
        for _bid in self.allBids:
            utility = get_utility(_bid, self.prefer, self.condition, self.domain_type, self.issue_value)
            if utility > lowerbound and utility < upperbound:
                bidsInRange.append(_bid)
        return bidsInRange

    def getBestBidByRouletteWheel(self, bidsInRange):
        size = len(bidsInRange)
        sumOfTwoUtilitiesForBid = [0.0] * size
        totalUtility = 0
        for i in range(size):
            bidDetails = bidsInRange[i]
            sum = self.opponentModel.getUtility(bidDetails)
            sumOfTwoUtilitiesForBid[i] = sum
            totalUtility += sum
        normalizedSumOfTwoUtilitiesForBid = []
        for i in range(size):
            if totalUtility == 0:
                normalizedSumOfTwoUtilitiesForBid.append(0)
            else:
                normalizedSumOfTwoUtilitiesForBid.append(sumOfTwoUtilitiesForBid[i] / totalUtility)
        random_ = random.random()
        integrate = 0
        selectedBidIndex = size-1
        for i in range(size):
            integrate += normalizedSumOfTwoUtilitiesForBid[i]
            if integrate >= random_:
                selectedBidIndex = i
                break
        return selectedBidIndex

    def getExplorableNeighbourhood(self):
        time = self.relative_t
        if time < self.timeToConcede:
            return 0
        else:
            return self.neigExplorationDisFactor * (1 - (self.pMin + (self.pMax - self.pMin) * (1 - self.f(time))))

    def tuneBidByOpponentModel(self, targetUtilityByTime):
        utilityThreshold = self.getExplorableNeighbourhood()
        # print('targetUtilityByTime:{} utilityThreshold:{}\n'.format(targetUtilityByTime, utilityThreshold))
        # print('lower:{} upper:{}\n'.format(targetUtilityByTime - utilityThreshold, targetUtilityByTime + utilityThreshold))
        bidsInRange = self.getBidsinRange(targetUtilityByTime - utilityThreshold, targetUtilityByTime + utilityThreshold)
        if len(bidsInRange) == 1:
            return bidsInRange[0]
        # FIXBUG
        if len(bidsInRange) == 0:
            return self.getBidNearUtility(targetUtilityByTime)
        selectedBidIndex = self.getBestBidByRouletteWheel(bidsInRange)
        # print('len(bidsInRange):{}\n'.format(len(bidsInRange)))
        # print('selectedBidIndex:{}\n'.format(selectedBidIndex))        
        return bidsInRange[selectedBidIndex]

    def getNextBid(self):
        time = self.relative_t
        targetUtilityByTime = self.getUtilityByTime(time)
        if targetUtilityByTime < self.minimumUtility:
            targetUtilityByTime = self.minimumUtility
        
        if self.isModelUsable() and self.canUseModel:
            return self.tuneBidByOpponentModel(targetUtilityByTime)
        else:
            bid = self.getBidNearUtility(targetUtilityByTime)
            return bid
