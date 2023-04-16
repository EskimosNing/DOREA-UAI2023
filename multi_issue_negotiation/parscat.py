import math
from agent import Agent
from utils import get_utility
import random


class ParsCat(Agent):
    def __init__(self, max_round, name="parscat agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.maxBid = None
        self.tresh = 0.0
        self.t1 = 0.0
        self.u2 = 1.0
        self.bestBidFromOpponent = None
        self.util_bestBidFromOpponent = 0.0

    def reset(self):
        super().reset()
        self.maxBid = self.bestBid
        self.tresh = 0.0
        self.t1 = 0.0
        self.u2 = 1.0
        self.bestBidFromOpponent = None
        self.util_bestBidFromOpponent = 0.0

    def receive(self, last_action=None):
        if last_action is not None:
            offer = last_action
            utility = get_utility(offer, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.utility_received.append(utility)
            self.offer_received.append(offer)
            self.u2 = utility
            self.t1 = math.ceil(self.t/2) / (self.max_round/2)
            if self.util_bestBidFromOpponent <= utility:
                self.bestBidFromOpponent = offer
                self.util_bestBidFromOpponent = utility


    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def getRandomBid(self):
        bid = None
        xxx = 0.001
        counter = 1000
        check = 0
        while counter == 1000:
            counter = 0
            while True:
                bid = [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]
                if self.t1 < 0.5:
                    self.tresh = 1 - self.t1 / 4
                    xxx = 0.01
                elif self.t1 >= 0.5 and self.t1 < 0.8:
                    self.tresh = 0.9 - self.t1 / 5
                    xxx = 0.02
                elif self.t1 >= 0.8 and self.t1 < 0.9:
                    self.tresh = 0.7 + self.t1 / 5
                    xxx = 0.02
                elif self.t1 >= 0.9 and self.t1 < 0.95:
                    self.tresh = 0.8 + self.t1 / 5
                    xxx = 0.02				
                elif self.t1 >= 0.95:
                    self.tresh = 1 - self.t1 / 4 - 0.01
                    xxx = 0.02
                if self.t1 == 1:
                    self.tresh = 0.5
                    xxx = 0.05
                self.tresh = self.tresh - check
                if self.tresh > 1:
                    self.tresh = 1
                    xxx = 0.01
                if self.tresh <= 0.5:
                    self.tresh = 0.49
                    xxx = 0.01
                counter = counter + 1

                util = get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)
               
                if (util < self.tresh - xxx or util > self.tresh + xxx) and (counter < 1000):
                    continue
                else:
                    break
            
            check = check + 0.01

        if self.getUtility(bid) < self.getUtility(self.bestBidFromOpponent):
            return self.bestBidFromOpponent
        
        return bid

    def gen_offer(self):
        if self.offer_received is None or len(self.offer_received) == 0:
            self.offer_proposed.append(self.maxBid)
            self.utility_proposed.append(self.getUtility(self.maxBid))
            return self.maxBid

        action = self.getRandomBid()
        myBid = action
        myOfferedUtil = self.getUtility(myBid)
        time = math.ceil(self.t/2) / (self.max_round/2)

        if len(self.offer_received) > 0 and self.offer_received[-1] == myBid:
            self.accept = True
            return None
        else:
            if self.offer_received is None or len(self.offer_received) == 0:
                OtherAgentBid = None
            else:
                OtherAgentBid = self.offer_received[-1]
            offeredUtilFromOpponent = self.getUtility(OtherAgentBid)
            if self.isAcceptable(offeredUtilFromOpponent, myOfferedUtil, time):
                self.accept = True
                return None
            else:
                self.offer_proposed.append(action)
                self.utility_proposed.append(myOfferedUtil)
                return action
        
    def isAcceptable(self, offeredUtilFromOtherAgent, myOfferedUtil, time):
        if offeredUtilFromOtherAgent == myOfferedUtil:
            return True

        Util = 1
        if time <= 0.25:
            Util = 1 - time * 0.4
        elif time > 0.25 and time <= 0.375:
            Util = 0.9 + (time - 0.25) * 0.4
        elif time > 0.375 and time <= 0.5:
            Util = 0.95 - (time - 0.375) * 0.4
        elif time > 0.5 and time <= 0.6:
            Util = 0.9 - (time - 0.5)
        elif time > 0.6 and time <= 0.7:
            Util = 0.8 + (time - 0.6) * 2
        elif time > 0.7 and time <= 0.8:
            Util = 1 - (time - 0.7) * 3
        elif time > 0.8 and time <= 0.9:
            Util = 0.7 + (time - 0.8)
        elif time > 0.9 and time <= 0.95:
            Util = 0.8 - (time - 0.9) * 6
        elif time > 0.95:
            Util = 0.5 + (time - 0.95) * 4
        if Util > 1:
            Util = 0.8

        return offeredUtilFromOtherAgent >= Util        
