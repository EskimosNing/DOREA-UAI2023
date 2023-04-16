from agent import Agent
from utils import get_utility
import random
import math
import copy
import numpy

class RandomDance(Agent):
    def __init__(self, max_round, name="RandomDance agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.NashCountMax = 200
        self.NumberOfAcceptSafety = 5
        self.NumberOfRandomTargetCheck = 3
        self.init = False
        self.utilityDatas = {} # Map<String, PlayerDataLib>
        self.myData = None  # PlayerData
        self.nash = [] # List<String>
        self.olderBidMap = {} # Map<String, Bid>
        self.discountFactor = 1.0
        self.reservationValue = 0
        self.olderTime = 0
        self.olderBid = None
        
        self.worstBid = []


    def reset(self):
        super().reset()
        self.NashCountMax = 200
        self.NumberOfAcceptSafety = 5
        self.NumberOfRandomTargetCheck = 3
        self.init = False
        self.utilityDatas = {}
        self.myData = None  # PlayerData
        self.nash = [] # List<String>
        self.olderBidMap = {} # Map<String, Bid>
        self.discountFactor = 1.0
        self.reservationValue = 0
        self.olderTime = 0
        self.olderBid = None
        self.worstBid = []
        for i in range(self.issue_num):
            min_value = 2
            min_key = None
            i_dict = self.issue_value[i]
            for key,value in i_dict.items():
                if value < min_value:
                    min_value = value
                    min_key = key
            self.worstBid.append(min_key)

    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def receive(self, last_action=None):
        if last_action is not None:
            utility = self.getUtility(last_action)
            self.offer_received.append(last_action)
            self.utility_received.append(utility)
            if 'oppo' not in self.utilityDatas.keys():
                self.utilityDatas['oppo'] = PlayerDataLib(self.issue_num, self.issue_value)
            self.olderBid = last_action
            self.olderBidMap['oppo'] = self.olderBid
            self.utilityDatas['oppo'].AddBid(self.olderBid)

    def myInit(self):
          playerData = PlayerData(self.issue_num, 1.0, self.issue_value)
          playerData.SetMyUtility(self.worstBid, self.getUtility(self.worstBid), self.issue_value, self.prefer)
          self.myData = playerData
          self.reservationValue = self.reservation
          self.discountFactor = self.discount
    
    def getWeights(self):
        playerWeight = {}  # Map<String, Double>
        rand = int(random.random() * 3)
        if rand == 0:
            for string in self.utilityDatas.keys():
                playerWeight[string] = 0.0001
            for string in self.nash:
                playerWeight[string] = playerWeight[string] + 1.0
        elif rand == 1:
            for string in self.utilityDatas.keys():
                playerWeight[string] = 1.0
        elif rand == 2:
            flag = random.random() < 0.5
            for string in self.utilityDatas.keys():
                if string == 'my':
                    continue
                if flag:
                    playerWeight[string] = 1.0
                else:
                    playerWeight[string] = 0.01
                flag = not flag
        else:
            for string in self.utilityDatas.keys():
                playerWeight[string] = 1.0
        return playerWeight

    def IsAccept(self, target, utility):
        time = self.relative_t
        d = time - self.olderTime
        self.olderTime = time
        if time + d * self.NumberOfAcceptSafety > 1.0:
            return True
        if self.olderBid is None:
            return False
        if utility > target:
            return True
        return False
    
    def IsEndNegotiation(self, target):
        if target < self.reservationValue:
            return True
        return False
    
    def GetTarget(self, datas):
        max_ = 0
        weights = {}
        for i in range(self.NumberOfRandomTargetCheck):
            utilityMap = {}
            for str in self.utilityDatas.keys():
                utilityMap[str] = self.utilityDatas[str].getRandomPlayerData()
                weights[str] = 1.0
            utilityMap['my'] = self.myData
            weights['my'] = 1.0
            bid = self.SearchBidWithWeights(utilityMap, weights)
            max_ = max(max_, self.getUtility(bid))
        # hear
        target = 1.0 - (1.0 - max_) * (pow(self.relative_t, self.discountFactor))
        if self.discountFactor > 0.99:
            target = 1.0 - (1.0 - max_) * (pow(self.relative_t, 3))
        return target

    def generateRandomBid(self):
        return [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]

    def SearchBidWithWeights(self, datas, weights):
        ret = self.generateRandomBid()
        idx = 0
        player0data = datas[list(datas.keys())[idx]]
        for i in range(self.issue_num):
            values = player0data.getIssueData(i).getValues()
            max_ = -1
            maxValue = None
            for value in values:
                v = 0
                for string in datas.keys():
                    data = datas[string]
                    weight = weights[string]
                    v += data.GetValue(i, value) * weight
                    # print(' v=', v)
                
                if v > max_:
                    max_ = v
                    maxValue = value
                # print('for value {}, v={}, max_={}'.format(value, v, max_))
            ret[i] = maxValue
        return ret

    def SearchBid(self, target, datas, weights):
        map = {}
        map = datas
        map['my'] = self.myData
        weightbuf = {}
        sum = 0
        for d in weights.values():
            sum += d
        for key in weights.keys():
            weightbuf[key] = weights[key] / sum
        for w in numpy.arange(0, 9.999, 0.01):
            if w == 1.0:
                continue
            myweight = w / (1.0 - w)
            weightbuf['my'] = myweight
            bid = self.SearchBidWithWeights(map, weightbuf)
            # print('bid:', bid)
            if self.getUtility(bid) > target:
                return bid
        return self.bestBid

    def gen_offer(self):
        if not self.init:
            self.init = True
            self.myInit()
        utilityMap = {} # Map<String, PlayerData>
        for str in self.utilityDatas.keys():
            utilityMap[str] = self.utilityDatas[str].getRandomPlayerData()
        utilityMap['my'] = self.myData
        # first bid => only 'my', otherwise 'my' + 'oppo'
        maxval = -999
        maxPlayer = None
        for str in self.olderBidMap.keys():
            utility = 1.0
            for player in utilityMap.keys():
                if str == player:
                    continue
                utility *= utilityMap[player].GetUtility(self.olderBidMap[str])
            if utility > maxval:
                maxval = utility
                maxPlayer = str
        if maxPlayer is not None:
            self.nash.append(maxPlayer)
        while len(self.nash) > self.NashCountMax:
            self.nash.remove(self.nash[0])
        playerWeight = self.getWeights()
        offer = None
        target = self.GetTarget(utilityMap)
        utility = 0
        if self.olderBid is not None:
            utility = self.getUtility(self.olderBid)
        offer = self.SearchBid(target, utilityMap, playerWeight)
        if offer is None or self.IsAccept(target, utility):
            self.accept = True
            return None
        if self.IsEndNegotiation(target):
            self.terminate = True
            return None
        offer_util = self.getUtility(offer)
        self.offer_proposed.append(offer)
        self.utility_proposed.append(offer_util)
        return offer


class IssueDataDiscrete:
    def __init__(self, issue_index, derta, issue_value):
        self.locked = False
        self.weight = 1
        self.derta = derta
        self.issue_index = issue_index
        self.max = 1
        self.map = {}  #Value,Double
        self.adder = 1.0
        self.issue_value = issue_value
        for value in self.getValues():
            self.setValue(value, 0)
    
    def getIssue(self):
        return self.issue_index

    def getValues(self):
        return list(self.issue_value[self.issue_index].keys())
        
    def Locked(self):
        self.locked = True
    
    def getWeight(self):
        return self.weight
    
    def setWeight(self, weight):
        self.weight = weight
    
    def isLocked(self):
        return self.locked
    
    def getMax(self):
        return self.max
    
    def GetValue(self, value):
        self.ValuePut(value)
        return self.map[value] / self.max
    
    def GetValueWithWeight(self, value):
        return self.GetValue(value) * self.getWeight()
    
    def Update(self, value):
        if self.isLocked():
            print('LockedAccess!')
            return
        self.ValuePut(value)
        self.map[value] = self.map[value] + self.adder
        self.max = max(self.max, self.map[value])
        self.adder *= self.derta
    
    def setValue(self, value, util):
        if self.isLocked():
            print("LockedAccess!!")
        else:
            self.map[value] = util

    def ValuePut(self, value):
        if value not in self.map:
            self.map[value] = 0.0

class PlayerData:
    def __init__(self, issue_num, derta, issue_value):
        self.issue_num = issue_num
        self.derta = derta
        self.map = [] #IssueData - Map<Issue, IssueData> map
        self.history = set() # Bid
        for i in range(self.issue_num):
            self.map.append(IssueDataDiscrete(i, derta, issue_value))
    
    def getIssueData(self, issue_index):
        return self.map[issue_index]

    def GetUtility(self, bid):
        ret = 0
        for i in range(self.issue_num):
            ret += self.GetValue(i, bid[i])
        return ret
    
    def GetValue(self, issue_index, value):
        return self.map[issue_index].GetValueWithWeight(value)
    
    def SetMyUtility(self, minBid, minUtility, issue_value, prefer):        
        for i in range(self.issue_num):
            issueData = self.map[i]
            bid = copy.deepcopy(minBid)
            values = issueData.getValues()
            for value in values:
                bid[i] = value
                v = get_utility(bid, prefer, 1, 'DISCRETE', issue_value) - minUtility
                issueData.setValue(value, v)
            issueData.setWeight(1.0/(1.0-minUtility))
            issueData.Locked()

    def AddBid(self, bid):
        if tuple(bid) in self.history:
            return
        self.history.add(tuple(bid))
        countsum = 0
        for i in range(self.issue_num):
            self.map[i].Update(bid[i])
            countsum += self.map[i].getMax()
        for i in range(self.issue_num):
            self.map[i].setWeight(self.map[i].getMax()/countsum)

class PlayerDataLib:
    def __init__(self, issue_num, issue_value):
        self.issue_num = issue_num
        self.playerDatas = []
        self.playerDatas.append(PlayerData(issue_num, 1.0, issue_value))
        self.playerDatas.append(PlayerData(issue_num, 1.05, issue_value))
        self.playerDatas.append(PlayerData(issue_num, 0.95, issue_value))

    def getRandomPlayerData(self):
        rand = int(random.random() * len(self.playerDatas))
        return self.playerDatas[rand]
    
    def AddBid(self, bid):
        for d in self.playerDatas:
            d.AddBid(bid)
    
    def getPlayerDataList(self):
        return self.playerDatas
