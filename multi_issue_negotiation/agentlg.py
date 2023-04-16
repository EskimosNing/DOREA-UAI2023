from numpy.lib.function_base import append
from agent import Agent
from utils import get_utility
from utils import get_utility_with_discount
import random
import math
import copy

def takeSecond(elem):
    return elem[1]

class BidStatistic:
    def __init__(self, issue):
        self.issue = issue
        self.valStatis = {}  # {value: integer}
        self.numVotes = 0

    def add(self, value):
        if self.valStatis.get(value) == None:
            self.valStatis[value] = 1
        else:
            self.valStatis[value] += 1
        self.numVotes += 1

    
    def getMostBided(self):
        maxval = 0
        maxtimes = 0
        for i in self.valStatis.keys():
            if self.valStatis.get(i) > maxtimes:
                maxtimes = self.valStatis.get(i)
                maxval = i
        return maxval, maxtimes

    def getValueUtility(self, value):
        ret = 0
        if self.valStatis.get(value) != None:
            maxval, maxtimes = self.getMostBided()
            ret = self.valStatis.get(value) / maxtimes
        return ret


class AgentLG(Agent):

    def __init__(self, max_round, name="AgentLG agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.t = 0
        self.relative_t = 0
        self.offer_received = []
        self.utility_received = []
        self.offer_proposed = []
        self.utility_proposed = []
        self.max_utility_received = 0
        self.max_offer_received = []  # the max bid for the agent from the opponent bids
        self.my_min_utility_bid = []
        self.my_min_utility_of_all_bids = 2
        self.bidLast = False
        self.allBids = None
        self.index = 0
        self.numPossibleBids = 0
        self.lastTimeLeft = 0
        self.maxLastOpponentBidUtility = 0
        # private HashMap<Issue, BidStatistic> statistic = new HashMap<Issue, BidStatistic>();
        self.statistic = {}


    def reset(self):
        super().reset()
        self.max_utility_received = 0
        self.max_offer_received = []
        self.my_min_utility_bid = []
        self.my_min_utility_of_all_bids = 2
        self.bidLast = False
        self.allBids = None
        # self.getAllBids()
        self.index = 0
        self.numPossibleBids = 0
        self.lastTimeLeft = 0
        self.maxLastOpponentBidUtility = 0
        self.statistic = {}
        for i in range(self.issue_num):
            self.statistic[i] = BidStatistic(i)

    def receive(self, last_action=None):
        if last_action is not None:
            offer = last_action
            utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
            if(self.max_utility_received < utility):
                self.max_utility_received = utility
                self.max_offer_received = offer
            self.utility_received.append(utility)
            self.offer_received.append(offer)

            if self.allBids is None or len(self.allBids) == 0:
                self.getAllBids()

			# updates statistics
            for i in range(self.issue_num):
                v = offer[i]
                self.statistic[i].add(v)

            # accept if opponent offer is good enough or there is no time and the offer is 'good'
            # if len(self.utility_proposed) != 0:
            #     if( (utility >=  self.utility_proposed[-1] * 0.99) or (((self.t+1)/self.max_round) > 0.999 and utility >=  self.utility_proposed[-1] * 0.9) or (self.getMyBidsMinUtility() <= utility) ):
            #         self.accept = True


    def getMyminBidfromBids(self):
        return self.allBids[self.numPossibleBids][0]

    def getMyBidsMinUtility(self, time):
        return get_utility_with_discount(self.allBids[self.numPossibleBids][0], self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount)

    # return all bids that utility > 1/4, and sorted by utility in descending order
    def getAllBids(self):
        _allBids = []
        issueValues = self.getIssueValues(issue_value_type = self.domain_type)
        firstValueBid = []
        for i in range(self.issue_num):
            firstValueBid.append(issueValues[i][0])
        _allBids.append(firstValueBid)

        for i in range(self.issue_num):
            tmpBids = []

            for bid in _allBids: # [[0,0,0]]
                tmpBid = bid
                for value in issueValues[i]:                  
                    tmpBid[i] = value
                    tmpBids.append(copy.deepcopy(tmpBid))
            
            _allBids = tmpBids

        # remove bids that are not good enough (the utility is less the 1/4 of the difference between the players)
        myBestUtility = self.u_max
        oppBestUtility = self.utility_received[0]

        # filteredBids is a list with tuple elements, each element is like ([issue1, issue2, issue3], utiltity)        
        filteredBids = self.filterBids(_allBids, myBestUtility, oppBestUtility, 0.75)

        # sort the filteredBids
        filteredBids.sort(key=takeSecond, reverse=True)
        self.allBids = filteredBids


    def filterBids(self, allBids, myBestUtility, oppBestUtility, fraction):
        downBond = myBestUtility - (myBestUtility - oppBestUtility)* fraction
        filteredBids = []

        for bid in allBids:
            bid_utility = get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)
            if bid_utility < downBond or bid_utility < self.u_min:
                continue
            else:
                bid_with_utility = (bid, bid_utility)
                filteredBids.append(bid_with_utility)
                if self.my_min_utility_of_all_bids > get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value):
                    self.my_min_utility_of_all_bids = get_utility(bid, self.prefer, self.condition, self.domain_type, self.issue_value)
                    self.my_min_utility_bid = bid
        # print("filteredBids = ", filteredBids)
        return filteredBids


    def getIssueValues(self, issue_value_type="REAL"):
        retvals = []
        if issue_value_type == "DISCRETE":
            for i in range(self.issue_num):
                retvals.append(list(self.issue_value[i].keys()))
        elif issue_value_type == "INTEGER":
            pass
        elif issue_value_type == "REAL":
            for i in range(self.issue_num):
                i_issue_value = []
                upperBound = 1.0
                lowerBound = 0.0
                intervalReal = (upperBound - lowerBound) / 10
                for i in range(11):
                    i_issue_value.append(lowerBound + i * intervalReal)
                retvals.append(i_issue_value)
        return retvals  # list, element i means i_th issue's all possible values,like ["dell", "lenova", "HP"]


    def gen_offer(self, offer_type="random"):
        if offer_type == "random":
            if len(self.offer_proposed) == 0 or len(self.offer_received) == 0:
                # first bid -> vote the optimal bid
                offer = self.bestBid
            else:
                opponentUtility = get_utility_with_discount(self.offer_received[-1], self.prefer, self.condition, self.domain_type, self.issue_value, self.relative_t, self.discount)
                myUtility = get_utility_with_discount(self.offer_proposed[-1], self.prefer, self.condition, self.domain_type, self.issue_value, self.relative_t, self.discount)
                if opponentUtility >= myUtility * 0.99 or (self.relative_t > 0.999 and opponentUtility >= myUtility * 0.9) or self.getMyBidsMinUtility(self.relative_t) <= opponentUtility :
                    self.accept = True
                    return None
                elif self.bidLast:
                    offer = self.offer_proposed[-1]
                elif self.relative_t < 0.6:            
                    offer = self.getNextOptimicalBid(self.relative_t)
                else:
                    if self.relative_t >= 0.9995:
                        offer = self.max_offer_received
                        if get_utility_with_discount(offer, self.prefer, self.condition, self.domain_type, self.issue_value, self.relative_t, self.discount) < self.reservation * math.pow(self.discount, self.relative_t):
                            offer = self.getMyminBidfromBids()
                    else:
                        offer = self.getNextBid(self.relative_t)
            
            if offer in self.offer_received:
                self.bidLast = True
            
            self.utility_proposed.append(get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value))
            self.offer_proposed.append(offer)
            return offer
        else:
            print("An error in gen_offer(): offer_type can only be random")
            exit(-1)

     
    """   
        Calculate the next optimal bid for the agent (from 1/4 most optimal bids)
    """
    def getNextOptimicalBid(self, time):
        if self.allBids is None or len(self.allBids) == 0:
            self.getAllBids()

        # print("self.allBids", self.allBids)
        bid = self.allBids[self.index][0]

        self.index += 1
        myBestUtility = get_utility_with_discount(self.bestBid, self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount)
        oppBestUtility = get_utility_with_discount(self.offer_received[0], self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount)
        downbound = myBestUtility - (myBestUtility - oppBestUtility) / 4

        # check if time passes and compromise a little bit
        if time - self.lastTimeLeft > 0.1 and self.numPossibleBids < len(self.allBids) - 1 and downbound <= get_utility_with_discount(self.allBids[self.numPossibleBids+1][0], self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount):
            futureUtility = get_utility_with_discount(self.allBids[self.numPossibleBids][0], self.prefer, self.condition, self.domain_type, self.issue_value, time + 0.1, self.discount)
            while get_utility_with_discount(self.allBids[self.numPossibleBids][0], self.prefer, self.condition, self.domain_type, self.issue_value, time, self.discount) >= futureUtility and self.numPossibleBids < len(self.allBids) - 1:                
                self.numPossibleBids += 1
            self.lastTimeLeft = time
        if self.index > self.numPossibleBids:
            self.index = 0
        self.maxLastOpponentBidUtility = self.max_utility_received

        return bid


    def getNextBid(self, time):
        bid = self.allBids[self.index][0]
        self.index += 1

        if self.index > self.numPossibleBids:
            # the time is over, compromising in a high rate
            if time >= 0.9:
                if time - self.lastTimeLeft > 0.008:
                    myBestUtility = get_utility(self.bestBid, self.prefer, self.condition, self.domain_type, self.issue_value)
                    oppBestUtility = self.utility_received[0]
                    avg = (myBestUtility + oppBestUtility) / 2
                    
                    # self.lastTimeLeft = self.relative_t
                    if self.index >= len(self.allBids):
                        self.index = len(self.allBids) - 1
                    elif self.allBids[self.index][1] < avg:
                        self.index -= 1                  
                        maxUtilty = 0
                        maxBidIndex = self.numPossibleBids
                        for i in range(self.numPossibleBids, self.index + 1):
                            # finds the next better bid for the opponent
                            utility = self.getOpponentBidUtility(self.allBids[i][0])
                            if utility > maxUtilty:
                                maxUtilty = utility
                                maxBidIndex = i
                        self.numPossibleBids = maxBidIndex
                    else:
                        self.index -= 1
                else:
                    self.index = 0
            else:
                self.index = 0
                # discount = 1.0
                # the time is over compromising in normal rate (0.05)
                if time - self.lastTimeLeft > 0.05:
                    # compromise only if the opponent is compromising
                    if (self.max_utility_received > self.maxLastOpponentBidUtility) or (self.discount < 1 and time - self.lastTimeLeft > 0.1):     
                        # finds the next better bid for the opponent
                        maxUtilty = 0
                        for i in range(self.numPossibleBids + 1):
                            utility = self.getOpponentBidUtility(self.allBids[i][0])
                            if utility > maxUtilty:
                                maxUtilty = utility
                        
                        for i in range(self.numPossibleBids + 1, len(self.allBids)):
                            utility = self.getOpponentBidUtility(self.allBids[i][0])
                            if utility > maxUtilty:
                                self.numPossibleBids = i
                                break
                        
                        self.maxLastOpponentBidUtility = self.max_utility_received
                        self.lastTimeLeft = time
        return bid


    # returns opponent bid utility that calculated from the vote statistics.
    def getOpponentBidUtility(self, bid):
        ret = 0
        for i in range(self.issue_num):
            ret += self.statistic.get(i).getValueUtility(bid[i])
        return ret
    