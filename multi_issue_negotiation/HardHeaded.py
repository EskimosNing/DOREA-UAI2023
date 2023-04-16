from agent import Agent
from utils import get_utility
import random
import math
import copy


class HardHeadedAgent(Agent):
    def __init__(self, max_round, name='hardheaded agent', u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.bidHistory = None
        self.BSelector = None
        self.MINIMUM_BID_UTILITY = 0.585
        self.TOP_SELECTED_BIDS = 4
        self.LEARNING_COEF = 0.2
        self.LEARNING_VALUE_ADDITION = 1
        self.UTILITY_TOLORANCE = 0.01
        self.Ka = 0.05
        self.e = 0.05
        self.discountF = 1
        self.lowestYetUtility = 1
        self.offerQueue = []
        self.opponentLastBid = None
        self.firstRound = True
        self.oppo_issue_weight = []
        self.oppo_issue_value = []
        self.oppo_issue_value_NotNormalized = []
        self.numberOfIssues = 0
        self.maxUtil = 1
        self.minUtil = self.MINIMUM_BID_UTILITY
        self.opponentbestbid = None
        self.opponentbestentry = None
        self.TEST_EQUIVALENCE = False
        self.round = 0


    def reset(self):
        super().reset()
        self.bidHistory = BidHistory(self.issue_num)
        self.BSelector = BidSelector(self.issue_num, self.issue_value, self.prefer)
        self.oppo_issue_value = copy.deepcopy(self.issue_value)
        self.oppo_issue_weight = copy.deepcopy(self.issue_weight)
        self.firstRound = True
        self.lowestYetUtility = 1
        self.offerQueue = []
        self.opponentLastBid = None
        self.numberOfIssues = self.issue_num
        if self.discount <= 1 and self.discount > 0:
            self.discountF = self.discount
        else:
            self.discountF = 1
        highestBid = self.BSelector.BidList.items()[-1] 
        self.maxUtil = get_utility(highestBid[1], self.prefer, self.condition, self.domain_type, self.issue_value)
        # get the number of issues and set a weight for each equal to 1/number_of_issues
        w = 1 / self.issue_num
        self.oppo_issue_weight = [w for i in range(self.issue_num)]
        # set the initial weight for each value of each issue to 1.
        for i in range(self.issue_num):
            for key in self.oppo_issue_value[i].keys():
                self.oppo_issue_value[i][key] = 1
        self.oppo_issue_value_NotNormalized = copy.deepcopy(self.oppo_issue_value)
        if self.reservation >= 0 and self.reservation <= 1:
            self.MINIMUM_BID_UTILITY = self.reservation
        else:
            self.MINIMUM_BID_UTILITY = 0.585
        self.minUtil = self.MINIMUM_BID_UTILITY
        self.opponentbestbid = None
        self.opponentbestentry = None
        self.round = 0

    def floorEntry(self, sortedDict, key):
        # (key, value)
        keySortedList = SortedList(sortedDict.keys())
        ret = keySortedList.bisect_right(key)
        if ret == 0:
            return None
        else:
            item_tuple = sortedDict.items()[ret-1]
            return item_tuple

    def lowerEntry(self, sortedDict, key):
        # (key, value)
        keySortedList = SortedList(sortedDict.keys())
        ret = keySortedList.bisect_left(key)
        if ret == 0:
            return None
        else:
            item_tuple = sortedDict.items()[ret-1]
            return item_tuple

    def receive(self, last_action):
        if last_action is not None:
            utility = get_utility(last_action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_received.append(utility)
            self.offer_received.append(last_action)
            self.opponentLastBid = last_action
            self.bidHistory.addOpponentBid(self.opponentLastBid)
            self.updateLearner()
            if self.opponentbestbid is None:
                self.opponentbestbid = self.opponentLastBid
            elif get_utility(self.opponentLastBid, self.prefer, self.condition, self.domain_type, self.issue_value) > get_utility(self.opponentbestbid, self.prefer, self.condition, self.domain_type, self.issue_value):
                self.opponentbestbid = self.opponentLastBid
            
            util_of_opponentbestbid = get_utility(self.opponentbestbid, self.prefer, self.condition, self.domain_type, self.issue_value)

            
            
            opbestvalue = self.floorEntry(self.BSelector.BidList, util_of_opponentbestbid)[0]
            
            while self.floorEntry(self.BSelector.BidList, opbestvalue)[1] != self.opponentbestbid:
                # print('self.floorEntry(self.BSelector.BidList, opbestvalue):', self.floorEntry(self.BSelector.BidList, opbestvalue))
                # print('self.opponentbestbid:', self.opponentbestbid)
                opbestvalue = self.lowerEntry(self.BSelector.BidList, opbestvalue)[0]
                
            self.opponentbestentry = self.floorEntry(self.BSelector.BidList, opbestvalue)

    def updateLearner(self):
        if self.bidHistory.getOpponentBidCount() < 2:
            return
        numberOfUnchanged = 0
        lastDiffSet = self.bidHistory.BidDifferenceofOpponentsLastTwo()  # HashMap<Integer, Integer> lastDiffSet
        # counting the number of unchanged issues
        for i in range(self.issue_num):
            if lastDiffSet[i] == 0:
                numberOfUnchanged += 1
        
        goldenValue = self.LEARNING_COEF / self.numberOfIssues
        totalSum = 1 + goldenValue * numberOfUnchanged
        maximumWeight = 1 - (self.numberOfIssues) * goldenValue / totalSum

        # re-weighing issues while making sure that the sum remains 1
        for i in range(self.issue_num):
            if lastDiffSet[i] == 0 and self.oppo_issue_weight[i] < maximumWeight:
                self.oppo_issue_weight[i] = (self.oppo_issue_weight[i] + goldenValue) / totalSum
            else:
                self.oppo_issue_weight[i] = self.oppo_issue_weight[i] / totalSum
        
        # Then for each issue value that has been offered last time, a constant value is added to its corresponding ValueDiscrete.
        for i in range(self.issue_num):
            self.oppo_issue_value_NotNormalized[i][self.opponentLastBid[i]] += self.LEARNING_VALUE_ADDITION

        self.normalizeOpponentIssueValue()

        
    def normalizeOpponentIssueValue(self):
        for i in range(self.issue_num):
            maxValue = -1
            for key in self.oppo_issue_value_NotNormalized[i].keys():
                if self.oppo_issue_value_NotNormalized[i][key] > maxValue:
                    maxValue = self.oppo_issue_value_NotNormalized[i][key]
            for key in self.oppo_issue_value_NotNormalized[i].keys():
                self.oppo_issue_value[i][key] = self.oppo_issue_value_NotNormalized[i][key] / maxValue


    # This function calculates the concession amount based on remaining time,initial parameters, and, the discount factor.
    def get_p(self):
        time = self.relative_t
        p = 1
        step_point = self.discountF
        tempMax = self.maxUtil
        tempMin = self.minUtil
        tempE = self.e
        ignoreDiscountThreshold = 0.9
        # Fa = 0

        if step_point >= ignoreDiscountThreshold:
            Fa = self.Ka + (1 - self.Ka) * math.pow(time / step_point, 1 / self.e)
            p = self.minUtil + (1 - Fa) * (self.maxUtil - self.minUtil)
        elif time <= step_point:
            tempE = self.e / step_point
            Fa = self.Ka + (1 - self.Ka) * math.pow(time / step_point, 1 / tempE)
            tempMin += abs(tempMax - tempMin) * step_point
            p = tempMin + (1 - Fa) * (tempMax - tempMin)
        else:
            tempE = 30
            Fa = (self.Ka + (1 - self.Ka) * math.pow((time - step_point) / (1 - step_point), 1 / tempE))
            tempMax = tempMin + abs(tempMax - tempMin) * step_point
            p = tempMin + (1 - Fa) * (tempMax - tempMin)

        return p


    """ 
    * This is the main strategy of that determines the behavior of the agent.
	 * It uses a concession function that in accord with remaining time decides
	 * which bids should be offered. Also using the learned opponent utility, it
	 * tries to offer more acceptable bids.	 * 
	 * @return {@link Action} that contains agents decision
    """
    def gen_offer(self):
        self.round += 1
        newBid = None  # Entry<Double, Bid> newBid
        p = self.get_p()

        if self.firstRound:
            self.firstRound = False
            newBid = self.BSelector.BidList.items()[-1]
            self.offerQueue.append(copy.deepcopy(newBid))
        elif self.offerQueue == None or len(self.offerQueue) == 0:
            newBids = SortedDict()
            newBid = self.lowerEntry(self.BSelector.BidList, self.bidHistory.getMyLastBid()[0])
            newBids[newBid[0]] = newBid[1]            
            if newBid[0] < p:
                indexer = self.bidHistory.getMyBidCount()
                indexer = int( math.floor(indexer * random.random()) )
                newBids.__delitem__(newBid[0])
                newBids[self.bidHistory.getMyBid(indexer)[0]] = self.bidHistory.getMyBid(indexer)[1]
            firstUtil = newBid[0]
            addBid = self.lowerEntry(self.BSelector.BidList, firstUtil)
            if addBid is None:
                addUtil = self.BSelector.BidList.keys()[0]
            else:
                addUtil = addBid[0]
            count = 0

            while ( (firstUtil - addUtil) < self.UTILITY_TOLORANCE ) and ( addUtil >= p ):
                if addBid is None:
                    break
                newBids[addUtil] = addBid[1]
                addBid = self.lowerEntry(self.BSelector.BidList, addUtil)
                addUtil = addBid[0]
                count += 1

            if len(newBids) <= self.TOP_SELECTED_BIDS:
                self.offerQueue.extend(list(newBids.items()))
            else:
                addedSofar = 0
                bestBid = None
                while addedSofar <= self.TOP_SELECTED_BIDS:
                    bestBid = newBids.items()[-1]
                    for e in newBids.items():
                        if get_utility(e[1], self.oppo_issue_weight, 1-self.condition, self.domain_type, self.oppo_issue_value) > get_utility(bestBid[1], self.oppo_issue_weight, 1-self.condition, self.domain_type, self.oppo_issue_value):
                            bestBid = e
                    self.offerQueue.append(copy.deepcopy(bestBid))
                    newBids.__delitem__(bestBid[0])
                    addedSofar += 1
            
            if self.offerQueue[0][0] < self.opponentbestentry[0]:
                self.offerQueue.insert(0, self.opponentbestentry)
        
        if self.offerQueue is None or len(self.offerQueue) == 0:
            bestBid1 = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
            if self.opponentLastBid is not None and get_utility(bestBid1, self.prefer, self.condition, self.domain_type, self.issue_value) <= get_utility(self.opponentLastBid, self.prefer, self.condition, self.domain_type, self.issue_value):
                self.accept = True
                return None
            elif bestBid1 is None:
                self.accept = True
                return None
            else:
                newAction = bestBid1
                if get_utility(bestBid1, self.prefer, self.condition, self.domain_type, self.issue_value) < self.lowestYetUtility:
                    self.lowestYetUtility = get_utility(bestBid1, self.prefer, self.condition, self.domain_type, self.issue_value)
        
        if (self.opponentLastBid is not None) and (get_utility(self.opponentLastBid, self.prefer, self.condition, self.domain_type, self.issue_value) > self.lowestYetUtility or  get_utility(self.offerQueue[0][1], self.prefer, self.condition, self.domain_type, self.issue_value) < get_utility(self.opponentLastBid, self.prefer, self.condition, self.domain_type, self.issue_value)):
            self.accept = True
            return None
        else:
            offer = self.offerQueue[0]
            self.offerQueue.__delitem__(0)
            self.bidHistory.addMyBid(offer)
            if offer[0] < self.lowestYetUtility:
                self.lowestYetUtility = get_utility(offer[1], self.prefer, self.condition, self.domain_type, self.issue_value)
            newAction = offer[1]
        
        newAction_util = get_utility(newAction, self.prefer, self.condition, self.domain_type, self.issue_value)
        self.offer_proposed.append(newAction)
        self.utility_proposed.append(newAction_util)
        return newAction


class BidHistory:
    def __init__(self, issue_num):
        self.issue_num = issue_num
        self.myBids = [] 
        self.opponentBids = []  
        

    def addMyBid(self, pBid):
        if pBid is None:
            print("pBid can't be null.")
            exit(0)
        self.myBids.append(pBid)
    
    def getMyBidCount(self):
        return len(self.myBids)

    def getMyBid(self, pIndex):
        return self.myBids[pIndex]

    def getMyLastBid(self):
        if self.getMyBidCount() > 0:
            return self.myBids[-1]
        else:
            return None
    
    def addOpponentBid(self, pBid):
        if pBid is None:
            print("pBid can't be null.")
            exit(0)
        self.opponentBids.append(pBid)

    def getOpponentBidCount(self):
        return len(self.opponentBids)

    def getOpponentBid(self, pIndex):
        return self.opponentBids[pIndex]

    def getOpponentLastBid(self):
        if self.getOpponentBidCount() > 0:
            return self.opponentBids[-1]
        else:
            return None

    def getOpponentSecondLastBid(self):
        if self.getOpponentBidCount() > 1:
            return self.opponentBids[-2]
        else:
            return None
    
    """ 
     * receives two bids as arguments and returns a {@link HashMap} that
	 * contains for each issue whether or not its value is different between the
	 * two bids.  RETURN - HashMap<Integer, Integer>
    """
    def BidDifference(self, first, second):
        diff = {}
        for i in range(self.issue_num):
            if first[i] == second[i]:
                isDiff = 0
            else:
                isDiff = 1
            diff[i] = isDiff
        return diff

    """ 
    * For the last two bids of the opponent returns a {@link HashMap} that
	 * contains for each issue whether or not its value is different between the
	 * two bids.
	 * 
	 * @return a {@link HashMap} with keys equal to issue IDs and with values 1
	 *         if different issue value observed and 0 if not.
    """
    def BidDifferenceofOpponentsLastTwo(self):
        if self.getOpponentBidCount() < 2:
            print("opponent bids less than two")
            exit(0)
        return self.BidDifference(self.getOpponentLastBid(), self.getOpponentSecondLastBid())


class BidSelector:
    def __init__(self, issue_num, issue_value, prefer):
        self.issue_num = issue_num
        self.issue_value = issue_value
        self.prefer = prefer
        self.BidList = SortedDict()  # new TreeMap<Double, Bid>()
        self.InitialBid = {}  # HashMap<Integer, Value> 
        for i in range(self.issue_num):
            self.InitialBid[i] = list(self.issue_value[i].keys())[0]
        b = list(self.InitialBid.values())
        self.BidList[get_utility(b, self.prefer, 1, 'DISCRETE', self.issue_value)] = b

        for i in range(self.issue_num):
            TempBids = SortedDict()  # TreeMap<Double, Bid> TempBids
            optionIndex  = len(self.issue_value[i])
            d = -0.00000001
            for TBid in self.BidList.values():
                for j in range(optionIndex):
                    NewBidV = self.Bidconfig(TBid)
                    NewBidV[i] = list(self.issue_value[i].keys())[j]
                    webid = list(NewBidV.values())
                    utility = get_utility(webid, self.prefer, 1, 'DISCRETE', self.issue_value)
                    TempBids[utility+d] = webid
                    d = d - 0.00000001
            self.BidList = TempBids
            
        

    def Bidconfig(self, pBid):
        lNewBidValues = SortedDict()
        for i in range(self.issue_num):
            lNewBidValues[i] = pBid[i]
        return lNewBidValues