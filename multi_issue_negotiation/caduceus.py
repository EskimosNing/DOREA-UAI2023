from agent import Agent
from utils import get_utility
import random
from ParsAgent import ParsAgent
from originalCaduceus import OriginalCaduceus
from atlas3 import Atlas3
from randomDance import RandomDance
from omacagent import OMAC


class Caduceus(Agent):

    def __init__(self, max_round, name="Caduceus agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round, name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.t = 0
        self.relative_t = 0
        self.offer_received = []
        self.utility_received = []
        self.offer_proposed = []
        self.utility_proposed = []
        self.discountFactor = 0
        self.selfReservationValue = 0.75
        self.percentageOfOfferingBestBid = 0.83
        self.agents = []
        self.scores = self.normalize([500, 10, 5, 3, 1])


    def reset(self):
        super().reset()
        self.agents = []
        parsagent = ParsAgent(max_round=self.max_round, name="ParsAgent")
        omac = OMAC(max_round=self.max_round)
        oriCaduceus = OriginalCaduceus(max_round=self.max_round, name="original caduceus agent")
        atlas3 = Atlas3(max_round=self.max_round, name="Atlas3 agent")
        randomDance = RandomDance(max_round=self.max_round)
        self.agents.append(parsagent)        
        self.agents.append(randomDance)
        self.agents.append(omac)
        self.agents.append(atlas3)
        self.agents.append(oriCaduceus)

        self.selfReservationValue = 0.75
        self.percentageOfOfferingBestBid = 0.83

        self.discountFactor = self.discount
        reservationValue = self.reservation
        self.selfReservationValue = max(self.selfReservationValue, reservationValue)
        self.percentageOfOfferingBestBid = self.percentageOfOfferingBestBid * self.discountFactor

        if self.domain_type == "DISCRETE":
            for i in range(len(self.agents)):             
                self.agents[i].issue_value = self.issue_value
                self.agents[i].issue_weight = self.issue_weight
                self.agents[i].prefer = self.prefer
                self.agents[i].bestBid = self.bestBid
                self.agents[i].discount = self.discount
                self.agents[i].reservation = self.reservation
                self.agents[i].condition = self.condition
                self.agents[i].issue_name = self.issue_name
                self.agents[i].bidSpace = self.bidSpace
                self.agents[i].issue_num = self.issue_num
                self.agents[i].oppo_prefer = self.oppo_prefer
                self.agents[i].domain_type = self.domain_type
                self.agents[i].reset()
        elif self.domain_type == "REAL":
            print("caduceus.py reset : i don't wanna do this part.")
            exit(-1)


    def receive(self, last_action=None):
        if last_action is not None:
            utility = get_utility(last_action, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.offer_received.append(last_action)
            self.utility_received.append(utility)
            for agent in self.agents:
                agent.receive(last_action)
    

    def gen_offer(self, offer_type="random"):
        if offer_type == "random":
            if self.isBestOfferTime():
                bestBid = self.bestBid
                utility = get_utility(bestBid, self.prefer, self.condition, self.domain_type, self.issue_value)
                self.offer_proposed.append(bestBid)
                self.utility_proposed.append(utility)
                return bestBid

            bidsFromAgents = []
            possibleActions = []
            for agent in self.agents:
                agent.t = self.t
                agent.relative_t = self.relative_t
                ret = agent.gen_offer()
                if ret == None and agent.accept == True:
                    possibleActions.append("accept")
                elif ret == None and agent.terminate == True:
                    possibleActions.append("terminate")
                else:
                    possibleActions.append(ret)

            scoreOfAccepts = 0
            scoreOfBids = 0
            agentsWithBids = []

            i = 0

            for bid in possibleActions:
                if bid == "accept":
                    scoreOfAccepts += self.getScore(i)
                elif isinstance(bid, list):
                    scoreOfBids += self.getScore(i)
                    bidsFromAgents.append(bid)
                    agentsWithBids.append(i)
                i += 1

            if scoreOfAccepts > scoreOfBids:
                self.accept = True
                return None
            elif scoreOfBids > scoreOfAccepts:
                retBid = self.getRandomizedAction(agentsWithBids, bidsFromAgents)
                if retBid is not None:
                    utility = get_utility(retBid, self.prefer, self.condition, self.domain_type, self.issue_value)
                    self.offer_proposed.append(retBid)
                    self.utility_proposed.append(utility)
                    return  retBid

            utility = get_utility(self.bestBid, self.prefer, self.condition, self.domain_type, self.issue_value)
            self.offer_proposed.append(self.bestBid)
            self.utility_proposed.append(utility)
            return self.bestBid


    def normalize(self, scores):
        normalized_scores = []
        sum = 0
        for i in range(len(scores)):
            sum += scores[i]
        for i in range(len(scores)):
            normalized_scores.append(scores[i]/sum)
        return normalized_scores

    def getScore(self, agentIndex):
        return self.scores[agentIndex]

    def isBestOfferTime(self):
        return self.t < self.max_round * self.percentageOfOfferingBestBid

    def getRandomizedAction(self, agentsWithBids, bidsFromAgents):
        possibilities = []
        # i = 0
        for agentWithBid in agentsWithBids:
            possibilities.append(self.getScore(agentWithBid))
        possibilities = self.normalize(possibilities)
        # print(possibilities)
        randomPick = random.random()
        acc = 0
        i = 0
        for possibility in possibilities:
            acc += possibility
            if randomPick < acc:
                return bidsFromAgents[i]
            i += 1
        
        return None
