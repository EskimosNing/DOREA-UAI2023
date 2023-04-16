from agent import Agent
from utils import get_utility
from utils import get_utility_with_discount
import random
import math


class CUHKAgent(Agent):

    def __init__(self, max_round, name="CUHK Agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.totalTime = max_round / 2
        self.ActionOfOpponent = None
        self.maximumOfBid = 0  # # bidSpace
        self.opponentBidHistory = OpponentBidHistory(self)
        self.minimumUtilityThreshold = 0
        self.utilitythreshold = u_max
        self.MaximumUtility = u_max
        self.timeLeftBefore = 0
        self.timeLeftAfter = 0
        self.maximumTimeOfOpponent = 0
        self.maximumTimeOfOwn = 0
        self.discountingFactor = 1
        self.concedeToDiscountingFactor = 0
        self.concedeToDiscountingFactor_original = 0
        self.minConcedeToDiscountingFactor = 0
        self.bidsBetweenUtility = None
        self.concedeToOpponent = False
        self.toughAgent = False
        self.alpha1 = 0.5
        self.bid_maximum_utility = []
        self.reservationValue = u_min
        self.fNumberOfDiscretizationSteps = 21
        self.minBidInOwnHistory = None 

    def reset(self):
        super().reset()
        self.ActionOfOpponent = None
        # self.ownBidHistory = OwnBidHistory()
        self.opponentBidHistory = OpponentBidHistory(self)
        if self.domain_type == 'REAL':
            self.maximumOfBid = 0x3F3F3F3F
            self.bid_maximum_utility = [self.condition] * self.issue_num
        elif self.domain_type == "DISCRETE":
            self.maximumOfBid = self.bidSpace
            self.bid_maximum_utility = self.bestBid
        self.bidsBetweenUtility = []
        self.utilitythreshold = 1
        self.MaximumUtility = self.utilitythreshold
        self.timeLeftBefore = 0
        self.timeLeftAfter = 0
        self.totalTime = self.max_round / 2
        self.maximumTimeOfOpponent = 0
        self.maximumTimeOfOwn = 0
        self.minConcedeToDiscountingFactor = 0.08
        self.discountingFactor = 1
        if self.discount <= 1.0 and self.discount > 0:
            self.discountingFactor = self.discount
        self.chooseUtilityThreshold()
        self.calculateBidsBetweenUtility()
        self.chooseConcedeToDiscountingDegree()
        self.opponentBidHistory.initializeDataStructures()
        self.timeLeftAfter = math.ceil(self.t/2)
        self.concedeToOpponent = False
        self.toughAgent = False
        self.alpha1 = 2
        self.minBidInOwnHistory = None
        self.reservationValue = 0
        if self.reservation > 0:
            self.reservationValue = self.reservation


    """ 
    Determine the lowest bound of our utility threshold based on the discounting factor. 
    We think that the minimum utility threshold should not be related with the discounting degree.
    """
    def chooseUtilityThreshold(self):
        if self.discountingFactor >= 0.9:
            self.minimumUtilityThreshold = 0
        else:
            self.minimumUtilityThreshold = 0
    

    """ pre-processing to save the computational time each round """
    def calculateBidsBetweenUtility(self):
        maximumUtility = self.MaximumUtility
        minUtility = self.minimumUtilityThreshold
        maximumRounds = int( (maximumUtility - minUtility) / 0.01 )  # Here, maximumRounds = 100
        # initalization for each arraylist storing the bids between each range
        for i in range(maximumRounds):
            BidList = []
            self.bidsBetweenUtility.append(BidList)
        self.bidsBetweenUtility[maximumRounds-1].append(self.bid_maximum_utility)
        #  note that here we may need to use some trick to reduce the
        #  computation cost (to be checked later);
        #  add those bids in each range into the corresponding arraylist
        limits = 0
        if self.maximumOfBid < 20000:
            allBids = self.getAllBids()
            # print(allBids)
            
            for bid in allBids:
                # print("bid :", bid)
                utility = get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)
                for i in range(maximumRounds):
                    if utility <= (i + 1) * 0.01 + minUtility and utility >= i * 0.01 + minUtility:
                        self.bidsBetweenUtility[i].append(bid)
                        break
        else:
            while limits <= 20000:
                bid = self.RandomSearchBid()
                utility = get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)
                for i in range(maximumRounds):
                    if utility <= (i + 1) * 0.01 + minUtility and utility >= i * 0.01 + minUtility:
                        self.bidsBetweenUtility[i].append(bid)
                        break
                limits += 1


    def RandomSearchBid(self):
        bid = []
        if self.domain_type == "REAL":
            for i in range(self.issue_num):
                optionInd = random.randint(0, self.fNumberOfDiscretizationSteps - 2)
                bid.append(0 + (1 - 0) * optionInd / self.fNumberOfDiscretizationSteps)
        elif self.domain_type == "DISCRETE":
            bid = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
        return bid


    # determine concede-to-time degree based on the discounting factor.
    def chooseConcedeToDiscountingDegree(self):
        alpha = 0
        beta = 1.5
        # the vaule of beta depends on the discounting factor
        if self.discountingFactor > 0.75:
            beta = 1.8
        elif self.discountingFactor > 0.5:
            beta = 1.5
        else:
            beta = 1.2
        alpha = math.pow(self.discountingFactor, beta)
        self.concedeToDiscountingFactor = self.minConcedeToDiscountingFactor + (1 - self.minConcedeToDiscountingFactor) * alpha
        self.concedeToDiscountingFactor_original = self.concedeToDiscountingFactor


    def receive(self, last_action=None):
        if last_action is not None:
            self.ActionOfOpponent = last_action
            utility = get_utility(last_action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_received.append(utility)
            self.offer_received.append(last_action)


    def gen_offer(self):

        self.timeLeftBefore = math.ceil(self.t/2)
        bid = []

        # we propose first and propose the bid with maximum utility
        if self.ActionOfOpponent is None:
            bid = self.bid_maximum_utility
        else:  # the opponent propose first and we response secondly
            self.opponentBidHistory.updateOpponentModel(self.ActionOfOpponent)
            self.updateConcedeDegree()
            if len(self.offer_proposed) == 0:
                bid = self.bid_maximum_utility
            else:
                if self.estimateRoundLeft(True) > 10:
                    bid = self.BidToOffer()
                    IsAccept = self.AcceptOpponentOffer(self.ActionOfOpponent, bid)
                    IsTerminate = self.TerminateCurrentNegotiation(bid)
                    if IsAccept == True and IsTerminate == False:
                        self.accept = True
                        return None  
                    elif IsAccept == False and IsTerminate == True:
                        self.terminate = True
                        return None
                    elif IsAccept == True and IsTerminate == True:
                        if get_utility(self.ActionOfOpponent, self.prefer, self.condition, self.domain_type, self.issue_value) > self.reservationValue:
                            self.accept = True
                            return None
                        else:
                            self.terminate = True
                            return None
                    else:
                        #  we expect that the negotiation is over once we select a bid from the opponent's history.
                        if self.concedeToOpponent == True:
                            bid = self.opponentBidHistory.getBestBidInHistory()
                            self.toughAgent = True
                            self.concedeToOpponent = False
                        else:
                            self.toughAgent = False
                else:
                    if math.ceil(self.t/2) / (self.max_round/2) > 0.9985 and self.estimateRoundLeft(True) < 5:
                        bid  = self.opponentBidHistory.getBestBidInHistory()
                        if get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value) < 0.85:
                            candidateBids = self.getBidsBetweenUtility(self.MaximumUtility-0.15, self.MaximumUtility-0.02)
                            if self.estimateRoundLeft(True) < 2:
                                bid = self.opponentBidHistory.getBestBidInHistory()
                            else:
                                bid = self.opponentBidHistory.ChooseBid(candidateBids)
                            if bid is None or len(bid) == 0:
                                bid = self.opponentBidHistory.getBestBidInHistory()
                        
                        IsAccept = self.AcceptOpponentOffer(self.ActionOfOpponent, bid)
                        IsTerminate = self.TerminateCurrentNegotiation(bid)
                        if IsAccept == True and IsTerminate == False:
                            self.accept = True
                            return None  
                        elif IsAccept == False and IsTerminate == True:
                            self.terminate = True
                            return None
                        elif IsAccept == True and IsTerminate == True:
                            if get_utility(self.ActionOfOpponent, self.prefer, self.condition, self.domain_type, self.issue_value) > self.reservationValue:
                                self.accept = True
                                return None
                            else:
                                self.terminate = True
                                return None
                        else: 
                            if self.toughAgent == True:
                                self.accept = True
                                return None

                    else:
                        bid = self.BidToOffer()
                        IsAccept = self.AcceptOpponentOffer(self.ActionOfOpponent, bid)
                        IsTerminate = self.TerminateCurrentNegotiation(bid)
                        if IsAccept == True and IsTerminate == False:
                            self.accept = True
                            return None  
                        elif IsAccept == False and IsTerminate == True:
                            self.terminate = True
                            return None
                        elif IsAccept == True and IsTerminate == True:
                            if get_utility(self.ActionOfOpponent, self.prefer, self.condition, self.domain_type, self.issue_value) > self.reservationValue:
                                self.accept = True
                                return None
                            else:
                                self.terminate = True
                                return None
                            
        self.offer_proposed.append(bid)
        self.utility_proposed.append(get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value))
        self.timeLeftAfter = math.ceil(self.t/2)
        self.estimateRoundLeft(False)

        return bid                           

    
    def TerminateCurrentNegotiation(self, ownBid):
        currentUtility = 0
        nextRoundUtility = 0
        maximumUtility = 0
        self.concedeToOpponent = False
        currentUtility = self.reservationValue
        nextRoundUtility = get_utility(ownBid, self.prefer, self.condition, self.domain_type, self.issue_value)
        maximumUtility = self.MaximumUtility

        if currentUtility >= self.utilitythreshold or currentUtility >= nextRoundUtility:
            return True
        else:
            # if the current reseravation utility with discount is larger than
			# the predicted maximum utility with discount then terminate the negotiation.
            predictMaximumUtility = maximumUtility * self.discountingFactor
            currentMaximumUtility = self.reservationValue * math.pow(self.discountingFactor, math.ceil(self.t/2) / (self.max_round/2))
            if currentMaximumUtility > predictMaximumUtility and math.ceil(self.t/2) / (self.max_round/2) > self.concedeToDiscountingFactor:
                return True
            else:
                return False


    # decide whether to accept the current offer or not
    def AcceptOpponentOffer(self, opponentBid, ownBid):
        currentUtility = 0
        nextRoundUtility = 0
        maximumUtility = 0
        self.concedeToOpponent = False
        currentUtility = get_utility(opponentBid, self.prefer, self.condition, self.domain_type, self.issue_value)
        maximumUtility = self.MaximumUtility
        nextRoundUtility = get_utility(ownBid, self.prefer, self.condition, self.domain_type, self.issue_value)

        if currentUtility >= self.utilitythreshold or currentUtility >= nextRoundUtility:
            return True
        else:
            # if the current utility with discount is larger than the predicted maximum utility with discount then accept it.
            predictMaximumUtility = maximumUtility * self.discountingFactor
            currentMaximumUtility = get_utility_with_discount(self.opponentBidHistory.getBestBidInHistory(), self.prefer, self.condition, self.domain_type, self.issue_value, math.ceil(self.t/2) / (self.max_round/2), self.discountingFactor)
            if currentMaximumUtility > predictMaximumUtility and math.ceil(self.t/2) / (self.max_round/2) > self.concedeToDiscountingFactor:                
                if get_utility(opponentBid, self.prefer, self.condition, self.domain_type, self.issue_value) >= get_utility(self.opponentBidHistory.getBestBidInHistory(), self.prefer, self.condition, self.domain_type, self.issue_value) - 0.01:
                    return True  # if the current offer is approximately as good as the best one in the history, then accept it. 
                else:
                    self.concedeToOpponent = True
                    return False
            elif currentMaximumUtility > self.utilitythreshold * math.pow(self.discountingFactor, math.ceil(self.t/2) / (self.max_round/2)):
                if get_utility(opponentBid, self.prefer, self.condition, self.domain_type, self.issue_value) >= get_utility(self.opponentBidHistory.getBestBidInHistory(), self.prefer, self.condition, self.domain_type, self.issue_value) - 0.01:
                    return True
                else:
                    self.concedeToOpponent = True
                    return False
            else:
                return False


    """ 
        principle: randomization over those candidate bids to let the opponent have a better model of my utility profile,
        return the bid to be offered in the next round
    """
    def BidToOffer(self):
        bidReturned = []
        decreasingAmount_1 = 0.05
        decreasingAmount_2 = 0.25
        maximumOfBid = self.MaximumUtility
        minimumOfBid = 0
        # used when the domain is very large. make concession when the domin is large
        if self.discountingFactor == 1 and self.maximumOfBid > 3000:
            minimumOfBid = self.MaximumUtility - decreasingAmount_1
            # make further concession when the deadline is approaching and the domain is large
            if self.discountingFactor > 1 - decreasingAmount_2 and self.maximumOfBid > 10000 and math.ceil(self.t/2) / (self.max_round/2) >= 0.98:
                minimumOfBid = self.MaximumUtility - decreasingAmount_2
            if self.utilitythreshold > minimumOfBid:
                self.utilitythreshold = minimumOfBid
        else:  # the general case
            if math.ceil(self.t/2) / (self.max_round/2) <= self.concedeToDiscountingFactor:
                minThreshold = (maximumOfBid * self.discountingFactor) / math.pow(self.discountingFactor, self.concedeToDiscountingFactor)
                self.utilitythreshold = maximumOfBid - (maximumOfBid - minThreshold) * math.pow((math.ceil(self.t/2) / (self.max_round/2) / self.concedeToDiscountingFactor), self.alpha1)
            else:
                self.utilitythreshold = (maximumOfBid * self.discountingFactor)	/ math.pow(self.discountingFactor, math.ceil(self.t/2) / (self.max_round/2))
            minimumOfBid = self.utilitythreshold
        
        # choose from the opponent bid history first to reduce calculation time
        bestBidOfferedByOpponent = self.opponentBidHistory.getBestBidInHistory()
        bestBidOfferedByOpponent_utility = get_utility(bestBidOfferedByOpponent, self.prefer, self.condition, self.domain_type, self.issue_value)
        if bestBidOfferedByOpponent_utility >= self.utilitythreshold or bestBidOfferedByOpponent_utility >= minimumOfBid:
            return bestBidOfferedByOpponent

        candidateBids = self.getBidsBetweenUtility(minimumOfBid, maximumOfBid)
        bidReturned = self.opponentBidHistory.ChooseBid(candidateBids)
        if bidReturned is None or len(bidReturned) == 0:
            bidReturned = self.bid_maximum_utility

        return bidReturned


    # Get all the bids within a given utility range.
    def getBidsBetweenUtility(self, lowerBound, upperBound):
        bidsInRange = []
        _range = int( (upperBound - self.minimumUtilityThreshold) / 0.01 )
        initial = int( (lowerBound - self.minimumUtilityThreshold) / 0.01 )
        for i in range(initial, _range):
            bidsInRange.extend(self.bidsBetweenUtility[i])
        if len(bidsInRange) == 0:
            bidsInRange.append(self.bid_maximum_utility)
        return bidsInRange


    # estimate the number of rounds left before reaching the deadline
    def estimateRoundLeft(self, opponent):
        round = 0
        if opponent == True:
            if self.timeLeftBefore - self.timeLeftAfter > self.maximumTimeOfOpponent:
                self.maximumTimeOfOpponent = self.timeLeftBefore - self.timeLeftAfter
        else:
            if self.timeLeftAfter - self.timeLeftBefore > self.maximumTimeOfOwn:
                self.maximumTimeOfOwn = self.timeLeftAfter - self.timeLeftBefore
        if self.maximumTimeOfOpponent + self.maximumTimeOfOwn == 0:
            return int(self.totalTime - math.ceil(self.t/2))
        round = (self.totalTime - math.ceil(self.t/2)) / (self.maximumTimeOfOpponent + self.maximumTimeOfOwn)  
        return int(round)


    # receiveMessage the concede-to-time degree based on the predicted toughness degree of the opponent
    def updateConcedeDegree(self):
        gama = 0
        weight = 0.1
        opponnetToughnessDegree = self.opponentBidHistory.getConcessionDegree()
        self.concedeToDiscountingFactor = self.concedeToDiscountingFactor_original + weight * (1 - self.concedeToDiscountingFactor_original) * math.pow(opponnetToughnessDegree, gama)
        if self.concedeToDiscountingFactor >= 1:
            self.concedeToDiscountingFactor = 1


class OpponentBidHistory:
    def __init__(self, agent):
        self.agent = agent
        self.bidHistory = []
        self.opponentBidsStatisticsForReal = []
        self.opponentBidsStatisticsDiscrete = []
        self.opponentBidsStatisticsForInteger = []
        self.maximumBidsStored = 100
        self.bidCounter = {}
        self.bid_maximum_from_opponent = []  # the bid with maximum utility proposed by the opponent so far.
        self.fNumberOfDiscretizationSteps = 21

    def initializeDataStructures(self):
        if self.agent.domain_type == 'REAL':
            for i in range(self.agent.issue_num):
                numProposalsPerValue = []
                lNumOfPossibleValuesInThisIssue = self.fNumberOfDiscretizationSteps
                for i in range(lNumOfPossibleValuesInThisIssue):
                    numProposalsPerValue.append(0)
                self.opponentBidsStatisticsForReal.append(numProposalsPerValue) 
        elif self.agent.domain_type == "DISCRETE":
            for i in range(self.agent.issue_num):
                discreteIssueValuesMap = {}
                for j in self.agent.issue_value[i].keys():
                    discreteIssueValuesMap[j] = 0
                self.opponentBidsStatisticsDiscrete.append(discreteIssueValuesMap)
        

    # This function updates the opponent's model by calling the updateStatistics method
    def updateOpponentModel(self, bidToUpdate):
        self.addBid(bidToUpdate)
        tuple_bidToUpdate = tuple(bidToUpdate)
        if tuple_bidToUpdate not in self.bidCounter:
            self.bidCounter[tuple_bidToUpdate] = 1
        else:
            counter = self.bidCounter[tuple_bidToUpdate]
            counter += 1
            self.bidCounter[tuple_bidToUpdate] = counter
        if len(self.bidHistory) <= self.maximumBidsStored:
            self.updateStatistics(bidToUpdate, False)

    def addBid(self, bid):
        if bid not in self.bidHistory:
            self.bidHistory.append(bid)
        if len(self.bidHistory) == 1:
            self.bid_maximum_from_opponent = self.bidHistory[0]
        else:
            if get_utility(bid, self.agent.prefer, self.agent.condition, self.agent.domain_type, self.agent.issue_value) > get_utility(self.bid_maximum_from_opponent, self.agent.prefer, self.agent.condition, self.agent.domain_type, self.agent.issue_value):
                self.bid_maximum_from_opponent = bid

    # This function updates the statistics of the bids that were received from the opponent.
    def updateStatistics(self, bidToUpdate, toRemove):
        if self.agent.domain_type == "DISCRETE":
            for i in range(self.agent.issue_num):
                if self.opponentBidsStatisticsDiscrete[i] != {}:
                    counterPerValue = self.opponentBidsStatisticsDiscrete[i][bidToUpdate[i]]
                    if toRemove:
                        counterPerValue -= 1
                    else:
                        counterPerValue += 1
                    self.opponentBidsStatisticsDiscrete[i][bidToUpdate[i]] = counterPerValue
        elif self.agent.domain_type == "REAL":
            for i in range(self.agent.issue_num):
                lOneStep = (1 - 0) / self.fNumberOfDiscretizationSteps
                first = 0
                last = first + lOneStep
                valueReal = bidToUpdate[i]
                found = False
                j = 0
                while j < len(self.opponentBidsStatisticsForReal[i]) and found == False:
                    if valueReal >= first and valueReal <= last:
                        counterPerValue = self.opponentBidsStatisticsForReal[i][j]
                        if toRemove:
                            counterPerValue -= 1
                        else:
                            counterPerValue += 1
                        self.opponentBidsStatisticsForReal[i][j] = counterPerValue
                        found = True
                    first = last
                    last = last + lOneStep
                    j += 1
                # If no matching value was found, receiveMessage the last cell
                if found == False:
                    k = len(self.opponentBidsStatisticsForReal[i]) - 1
                    counterPerValue = self.opponentBidsStatisticsForReal[i][k]
                    if toRemove:
                        counterPerValue -= 1
                    else:
                        counterPerValue += 1
                    self.opponentBidsStatisticsForReal[i][k] = counterPerValue

    # Another way to predict the opponent's concession degree
    def getConcessionDegree(self):
        numOfBids = len(self.bidHistory)
        numOfDistinctBid = 0
        historyLength = 10
        concessionDegree = 0
        if numOfBids - historyLength > 0:
            for j in range(numOfBids - historyLength, numOfBids):  
                if self.bidCounter[tuple(self.bidHistory[j])] == 1:
                    numOfDistinctBid += 1
            concessionDegree = math.pow(numOfDistinctBid / historyLength, 2)  
        else:
            numOfDistinctBid = self.getSize()
            concessionDegree = math.pow(numOfDistinctBid / historyLength, 2)
        return concessionDegree
    
    def getSize(self):
        numOfBids = len(self.bidHistory)
        # print("bidHistory : ", self.bidHistory)
        # print("NUMBER OF distinct bid : ", numOfBids)
        return numOfBids

    def getBestBidInHistory(self):
        return self.bid_maximum_from_opponent

    # choose a bid which is optimal for the opponent among a set of candidate bids.
    def ChooseBid(self, candidateBids):
        upperSearchLimit = 200
        maxIndex = -1
        maxFrequency = 0

        if len(candidateBids) >= upperSearchLimit:
            bids = []
            for i in range(upperSearchLimit):
                ranIndex = random.randint(0, len(candidateBids)-1)
                bids.append(candidateBids[ranIndex])
            candidateBids = bids
        
        if self.agent.domain_type == "DISCRETE":
            # this whole block of code is to find the best bid
            for i in range(len(candidateBids)):
                maxValue = 0  
                for j in range(self.agent.issue_num):
                    maxValue += self.opponentBidsStatisticsDiscrete[j][candidateBids[i][j]]
                if maxValue > maxFrequency:
                    maxFrequency = maxValue
                    maxIndex = i
                elif maxValue == maxFrequency:
                    if random.random() < 0.5:
                        maxIndex = i
        elif self.agent.domain_type == "REAL":
            for i in range(len(candidateBids)):
                maxValue = 0
                for j in range(self.agent.issue_num):
                    lNumOfPossibleRealValues = self.fNumberOfDiscretizationSteps
                    lOneStep = (1 - 0) / lNumOfPossibleRealValues
                    first = 0
                    last = first + lOneStep
                    valueReal = candidateBids[i][j]
                    found = False
                    k = 0
                    while found == False and k < len(self.opponentBidsStatisticsForReal[j]):
                        if valueReal >= first and valueReal <= last:
                            maxValue += self.opponentBidsStatisticsForReal[j][k]
                            found = True
                        first = last
                        last = last + lOneStep
                    if found == False:
                        maxValue += self.opponentBidsStatisticsForReal[j][-1]
                if maxValue > maxFrequency:
                    maxFrequency = maxValue
                    maxIndex = i
                elif maxValue == maxFrequency:
                    if random.random() < 0.5:
                        maxIndex = i
        
        if maxIndex == -1:
            return candidateBids[random.randint(0, len(candidateBids)-1)]
        else:
            # // here we adopt the random exploration mechanism
            if random.random() < 0.95:
                return candidateBids[maxIndex]
            else:
                return candidateBids[random.randint(0, len(candidateBids)-1)]
