from agent import Agent
from utils import get_utility, get_utility_with_discount
import random
import math
import copy
from functools import reduce


class UtilFunctions:
    @staticmethod
    def subtract(a_list, b_list):
        assert len(a_list) == len(b_list)
        result = [0] * len(a_list)
        for i in range(len(a_list)):
            result[i] = a_list[i] - b_list[i]
        return result

    @staticmethod
    def equals(a_list, b_list, precision):
        assert len(a_list) == len(b_list)
        distance = UtilFunctions.subtract(a_list, b_list)
        for i in range(len(a_list)):
            if distance[i] > precision:
                return False
        return True
    
    @staticmethod
    def getEuclideanDistance(a_list, b_list):
        assert len(a_list) == len(b_list)
        distance = 0
        for i in range(len(a_list)):
            d = a_list[i] - b_list[i]
            distance += d * d
        return math.sqrt(distance)

    @staticmethod
    def calculateUnitVector(a_list, b_list):
        assert len(a_list) == len(b_list)
        if a_list == b_list:
            unitVector = a_list
        else:
            unitVector = UtilFunctions.subtract(b_list, a_list)        
        norm = UtilFunctions.norm(unitVector)
        unitVector = UtilFunctions.divide(unitVector, norm)
        return unitVector
    
    @staticmethod
    def norm(a_list):
        norm = 0
        for i in range(len(a_list)):
            norm += a_list[i] * a_list[i]
        norm = math.sqrt(norm)
        return norm
    
    @staticmethod
    def divide(a_list, number):
        result = [0] * len(a_list)
        for i in range(len(a_list)):
            result[i] = a_list[i] / number
        return result

    @staticmethod
    def add(a_list, b_list):
        assert len(a_list) == len(b_list)
        result = [0] * len(a_list)
        for i in range(len(a_list)):
            result[i] = a_list[i] + b_list[i]
        return result
    
    @staticmethod
    def multiply(a_list, number):
        result = [0] * len(a_list)
        for i in range(len(a_list)):
            result[i] = a_list[i] * number
        return result

    @staticmethod
    def normalize(a_list):
        result = [0] * len(a_list)
        sum = 0
        for i in range(len(a_list)):
            result[i] = a_list[i]
            sum += a_list[i]
        if sum == 0:
            sum = 1

        for i in range(len(a_list)):
            result[i] = result[i] / sum
        return result

class NashProductCalculator:
    # def __init__(self, issue_num, issue_value, prefer, oppo_issue_value, oppo_prefer, allBids):
    def __init__(self, agent):
        self.nashProduct = 0.0
        self.nashBid = None
        self.issue_num = agent.issue_num
        self.issue_value = agent.issue_value
        self.prefer = agent.prefer
        self.oppo_issue_value = agent.oppo_issue_value
        self.oppo_prefer = agent.oppo_prefer
        self.allBids = agent.allBids

        if self.oppo_issue_value is not None and self.oppo_prefer is not None:
            for i in range(self.issue_num):
                maxValue = 0.0
                for key in self.oppo_issue_value[i].keys():                    
                    if self.oppo_issue_value[i][key] > maxValue:
                        maxValue = self.oppo_issue_value[i][key]
                for key in self.oppo_issue_value[i].keys():
                    self.oppo_issue_value[i][key] = self.oppo_issue_value[i][key] / maxValue
            sum = 0
            for i in range(self.issue_num):
                sum += self.oppo_prefer[i]
            for i in range(self.issue_num):
                self.oppo_prefer[i] /= sum

    
    def calculate(self):
        tempProduct = 1
        count = 0
        for bid in self.allBids:
            u1 = get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)
            if self.oppo_prefer is None and self.oppo_issue_value is None:
                u2 = 1
            else:
                u2 = get_utility(bid, self.oppo_prefer, 0, 'DISCRETE', self.oppo_issue_value)
            tempProduct = u1 * u2
            if tempProduct == 1.0:
                count += 1
            if tempProduct > self.nashProduct:
                self.nashProduct = tempProduct
                self.nashBid = bid

class CounterOfferGenerator:
    NUMBER_OF_ROUNDS_FOR_CONCESSION = 10
    # def __init__(self, nashBid, issue_num, issue_value, prefer, discount, bestBid, allBids):
    def __init__(self, nashBid, agent):
        self.allPossibleBids = agent.allBids
        self.nashBid = nashBid
        self.issue_num = agent.issue_num
        self.issue_value = agent.issue_value
        self.maxBid = agent.bestBid
        self.prefer = agent.prefer
        self.discount = agent.discount
        self.concessionStep = 0.2
        self.vectorSize = self.issue_num
        # self.calculateAllPossibleBids()
        self.bidSpace = []
        for i in range(len(self.allPossibleBids)):
            eachEle = [0.0]*self.vectorSize
            self.bidSpace.append(eachEle)
        self.vectorizeAll()
        self.concessionStep = (1.0 - self.getUtility(self.nashBid)) / CounterOfferGenerator.NUMBER_OF_ROUNDS_FOR_CONCESSION


    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, 'DISCRETE', self.issue_value)

    def calculateAllPossibleBids(self):
        self.allPossibleBids = self.getAllBids()

    def vectorizeAll(self):
        index = 0
        for bid in self.allPossibleBids:
            point = self.vectorizeBid(bid)
            self.bidSpace[index] = point
            index += 1
    
    # bid => double[]
    def vectorizeBid(self, bid):
        point = [0] * self.vectorSize
        issueIndex = 0
        for i in range(self.issue_num):
            point[issueIndex] = self.prefer[i] * self.issue_value[i][bid[i]]
            issueIndex += 1
        point = UtilFunctions.normalize(point)
        point = UtilFunctions.multiply(point, 10)
        return point
    
    def lists_combination(self, lists, code=''):
            
        def myfunc(list1, list2):
            return [str(i)+code+str(j) for i in list1 for j in list2]

        return reduce(myfunc, lists)

    def getAllBids(self):
        _allBids = []
        issueValues = self.getIssueValues()
        resList = self.lists_combination(issueValues,'@')
        for i in range(len(resList)):
            tmpList = resList[i].split('@')
            _allBids.append(tmpList)
        return _allBids

    def getIssueValues(self):
        retvals = []          
        for i in range(self.issue_num):
            retvals.append(list(self.issue_value[i].keys()))
        return retvals  # list, element i means i_th issue's all possible values,like ["dell", "lenova", "HP"]

    # return double[]
    def getUnitVector(self):
        maxBid = copy.deepcopy(self.maxBid)
        maxBidPoint = self.vectorizeBid(maxBid)
        nashPoint = self.vectorizeBid(self.nashBid)
        unitVector = UtilFunctions.calculateUnitVector(maxBidPoint, nashPoint)
        return unitVector

    def generateBid(self, concessionRate):
        maxBid = copy.deepcopy(self.maxBid)
        maxBidPoint = self.vectorizeBid(maxBid)
        delta = concessionRate
        unitVector = self.getUnitVector()
        concessionDelta = UtilFunctions.multiply(unitVector, delta)
        concessionPoint = UtilFunctions.add(maxBidPoint, concessionDelta)
        bid = self.getBidCloseToConcessionPoint(concessionPoint)
        return bid
    
    def getBidCloseToConcessionPoint(self, concessionPoint):
        maxBid = copy.deepcopy(self.maxBid)
        maxBidPoint = self.vectorizeBid(maxBid)
        distances = [0.0] * len(self.bidSpace)
        for i in range(len(distances)):
            bidPoint = self.bidSpace[i]
            distances[i] = UtilFunctions.getEuclideanDistance(concessionPoint, bidPoint)
        minDistance = distances[0]
        minDistanceIndex = 0
        for i in range(len(distances)):
            d = distances[i]
            if not UtilFunctions.equals(self.bidSpace[i], maxBidPoint, 0.1) and d < minDistance:
                minDistanceIndex = i
                minDistance = d
        bid = self.allPossibleBids[minDistanceIndex]
        return bid

class OriginalCaduceus(Agent):
    def __init__(self, max_round, name="original caduceus agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.opponentBidHistory = []
        self.discountFactor = 0
        self.selfReservationValue = 0.75
        self.percentageOfOfferingBestBid = 0.83
        self.previousBid = None
        self.takeConcessionStep = True
        
    def reset(self):
        super().reset()
        self.opponentBidHistory = []
        self.discountFactor = self.discount
        reservationValue = self.reservation
        self.selfReservationValue = max(self.selfReservationValue, reservationValue)
        self.percentageOfOfferingBestBid = self.percentageOfOfferingBestBid * self.discountFactor
        self.oppo_issue_value = None
        self.oppo_prefer = None
        self.previousBid = None
        self.takeConcessionStep = True

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def getRandomBid(self):
        while True:
            offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
            if self.getUtility(offer) > self.selfReservationValue:
                break
        return offer

    def receive(self, last_action=None):
        if last_action is not None:
            uglyBid = last_action
            uglyBid_u = self.getUtility(uglyBid)
            self.offer_received.append(uglyBid)
            self.utility_received.append(uglyBid_u)
            self.previousBid = uglyBid
            if self.oppo_prefer is None:
                self.oppo_prefer = copy.deepcopy(self.prefer)
                self.oppo_issue_value = copy.deepcopy(self.issue_value)
            previousBid = None
            if self.opponentBidHistory is not None and len(self.opponentBidHistory) != 0:
                previousBid = self.opponentBidHistory[-1]
            for i in range(self.issue_num):
                self.oppo_issue_value[i][uglyBid[i]] += self.getRoundValue()
                if previousBid is not None:
                    for j in range(self.issue_num):
                        previousBidValue = previousBid[j]
                        if previousBidValue == uglyBid[i]:
                            self.oppo_prefer[i] += self.getRoundValue()
            self.opponentBidHistory.append(uglyBid)
    
    def getRoundValue(self):
        roundValue = (2 * math.pow(self.relative_t, 2)) - (101 * self.relative_t) + 100
        return roundValue

    def isBestOfferTime(self):
        return self.relative_t < self.percentageOfOfferingBestBid

    def getMyBestOfferForEveryone(self, time):
        if self.allBids is None:
            self.allBids = self.getAllBids()
        npc = NashProductCalculator(self)
        npc.calculate()
        if npc.nashBid is None or len(npc.nashBid) == 0:
            bestBid = copy.deepcopy(self.bestBid)
            offerGenerator = CounterOfferGenerator(bestBid, self)
            return offerGenerator.generateBid(time)
        cog = CounterOfferGenerator(npc.nashBid, self)
        return cog.generateBid(time)

    def getUtilityWithDiscount(self, bid, time):
        return get_utility_with_discount(bid, self.prefer, 1, 'DISCRETE', self.issue_value, time, self.discount)

    def gen_offer(self):
        if self.isBestOfferTime():
            bestBid = self.bestBid
            best_u = self.getUtility(bestBid)
            self.offer_proposed.append(bestBid)
            self.utility_proposed.append(best_u)
            return bestBid
        else:
            bid = self.getMyBestOfferForEveryone(self.relative_t)
            if bid is not None:
                if self.getUtilityWithDiscount(bid, self.relative_t) < self.selfReservationValue:
                    bid = self.getRandomBid()
                if self.getUtilityWithDiscount(self.previousBid, self.relative_t) > self.getUtilityWithDiscount(bid, self.relative_t) + 0.2:
                    self.accept = True
                    return None
            else:
                print('bid is None. Something wrong.')
                exit(-1)
            bid_u = self.getUtility(bid)
            self.offer_proposed.append(bid)
            self.utility_proposed.append(bid_u)
            return bid
                    