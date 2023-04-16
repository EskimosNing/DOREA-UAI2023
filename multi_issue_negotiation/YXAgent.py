from agent import Agent
from utils import get_utility
import random
import math
import copy

class YXAgent(Agent):
    def __init__(self, max_round, name="YXAgent agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.myAction = None
        self.myLastBid = None
        self.lastOpponentBid = None
        self.myUtility = 0
        self.oppUtility = 0
        self.rv = u_min
        self.discountFactor = 0
        self.rounds = 0
        self.updatedValueIntegerWeight = False
        self.issueContainIntegerType = False
        self.searchedDiscountWithRV = False
        self.oppIssueWeight = []
        self.oppBidHistory = []
        self.oppIssueIntegerValue = []
        self.oppValueFrequency = []   
        self.oppSumDiff = 0  # double
        self.startTime = 0
        self.currTime = 0
        self.diff = 0
        self.totalTime = self.max_round
        self.normalizedTime = 0
        self.terminate = False


    def reset(self):        
        super().reset()
        self.myAction = None
        self.myLastBid = None
        self.lastOpponentBid = None
        self.myUtility = 0
        self.oppUtility = 0
        self.rv = self.reservation
        self.discountFactor = self.discount
        self.oppBidHistory = []
        self.oppIssueWeight = []
        self.oppIssueIntegerValue = []
        self.oppValueFrequency = []
        self.oppSumDiff = 0
        self.rounds = 0
        self.currTime = 0
        self.normalizedTime = 0
        self.terminate = False
        self.updatedValueIntegerWeight = False
        self.issueContainIntegerType = False
        self.searchedDiscountWithRV = False
        self.initTime()
        self.initOpp()

    def initTime(self):
        self.startTime = self.t
        self.totalTime = self.max_round
        self.diff = 0

    def initOpp(self):
        # init opponent issue weight
        avgW = 1 / self.issue_num
        self.oppIssueWeight = [avgW for _ in range(self.issue_num)]
        # init opponent value frequency
        self.oppValueFrequency = copy.deepcopy(self.issue_value)
        for i in range(self.issue_num):
            for key in self.oppValueFrequency[i].keys():
                self.oppValueFrequency[i][key] = 1
        
    def updateOpp(self):
        # if self.rounds <= 10:
        #     self.updateValueIntegerWeight()
        self.updateModelOppIssueWeight()
        self.updateModelOppValueWeight()
        self.oppBidHistory.append(self.lastOpponentBid)

    def updateModelOppIssueWeight(self):
        if len(self.oppBidHistory) != 0 and self.rounds >= 10:
            previousRoundBid = self.oppBidHistory[-1]
            issueWeightFormula = (math.pow((1 - self.normalizedTime), 10)) / (self.issue_num * 100)
            issueSum = 0
            for i in range(self.issue_num):
                prevIssueValue = previousRoundBid[i]
                currIssueValue = self.lastOpponentBid[i]
                # for only discrete domain
                if prevIssueValue == currIssueValue:
                    self.oppIssueWeight[i] = issueWeightFormula +  self.oppIssueWeight[i]
                issueSum += self.oppIssueWeight[i]
            # After Sum computed, Normalized Issue Weight
            for i in range(self.issue_num):
                self.oppIssueWeight[i] = self.oppIssueWeight[i] / issueSum

    def updateModelOppValueWeight(self):
        valueWeightFormula = math.pow(0.2, self.normalizedTime) / 30000
        valueWeightInteger = math.pow(0.2, self.normalizedTime) / 955
        for i in range(self.issue_num):
            value = self.lastOpponentBid[i]
            self.oppValueFrequency[i][value] = self.oppValueFrequency[i][value] + valueWeightFormula
        
        # Normalization of Value Weight
        for i in range(self.issue_num):
            maxValueBase = 0
            # Compute Max Value for specific issue
            for key in self.oppValueFrequency[i].keys():
                currValueBase = self.oppValueFrequency[i][key]
                if currValueBase > maxValueBase:
                    maxValueBase = currValueBase
            
            # After Max Value computed, Normalized Value Weight
            for key in self.oppValueFrequency[i].keys():
                self.oppValueFrequency[i][key] = self.oppValueFrequency[i][key] / maxValueBase


    def updateValueIntegerWeight(self):
        pass

    def updateTime(self):
        self.currTime = self.t
        self.diff = self.currTime - self.startTime
        self.normalizedTime = self.diff / self.totalTime

    def generateRandomBid(self):
        offer = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        return offer

    # Check Discount Factor wrt Reservation Value
    def evaluateDiscountFactorNReservationValue(self):
        init = 1.0
        initRV1 = 0.75
        initRV2 = 0.86
        initRV3 = 0.95
        selfDiscount = 0.998
        deduction = 0.005
        endNegotiation = False

        i = init
        while(i >= 0):
            if self.discountFactor == i and self.rv >= initRV3:
                endNegotiation = True
            initRV3 = (initRV3 - deduction) * selfDiscount        
            i -= 0.01
        self.searchedDiscountWithRV = True

        return endNegotiation


    def receive(self, last_action):
        if last_action is not None:
            # self.rounds += 1
            self.lastOpponentBid = last_action
            self.oppUtility = get_utility(self.lastOpponentBid, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.offer_received.append(last_action)
            self.utility_received.append(self.oppUtility)
            self.updateOpp()


    def gen_offer(self):
        self.updateTime()
        self.rounds += 1
        testBid = None
        calUtil = 0
        minimalThreshold = 0.7
        deductThreshold = 0.1
        calculatedThreshold = 1 - deductThreshold
        tempThreshold = max(minimalThreshold, calculatedThreshold)
        tempThreshold = max(tempThreshold, self.rv)

        # YXAgent Start 1st
        if len(self.oppBidHistory) == 0:
            while True:
                self.myLastBid = self.generateRandomBid()
                self.myUtility = get_utility(self.myLastBid, self.prefer, self.condition, self.domain_type, self.issue_value)
                if self.myUtility >= minimalThreshold:
                    break
            self.offer_proposed.append(self.myLastBid)
            self.utility_proposed.append(self.myUtility)
            return self.myLastBid

        # YXAgent Start 2nd
        if self.searchedDiscountWithRV == False and self.rounds > 1:
            if self.evaluateDiscountFactorNReservationValue() == True:
                self.terminate = True
                return None

        while(True):
            testBid = self.generateRandomBid()
            self.myUtility = get_utility(testBid, self.prefer, self.condition, self.domain_type, self.issue_value)
            if self.myUtility >= tempThreshold:
                break

        # Acceptance Criteria
        if self.rounds > 10 and self.normalizedTime <= 0.9:
            for i in range(self.issue_num):
                v = self.lastOpponentBid[i]
                calUtil += self.oppIssueWeight[i] * self.oppValueFrequency[i][v]
            
            calThreshold = calUtil - deductThreshold * 3 / 4
            calThreshold = max(tempThreshold, calThreshold)
            if self.oppUtility > calThreshold:
                self.accept = True
                return None
            else:
                self.offer_proposed.append(testBid)
                self.utility_proposed.append(self.myUtility)
                return testBid
        else:
            if self.oppUtility > tempThreshold:
                self.accept = True
                return None
            else:
                self.offer_proposed.append(testBid)
                self.utility_proposed.append(self.myUtility)
                return testBid
