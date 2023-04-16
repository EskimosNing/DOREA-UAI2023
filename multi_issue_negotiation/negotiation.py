import math
import random
from utils import get_utility
import xml.dom.minidom
import copy
from functools import reduce


class Negotiation:
    '''
    a negotiation environment to simulating buyer and seller setting in continuos domain.
    '''
    def __init__(self, max_round=30, issue_num=3, render=False, domain_type="DISCRETE", domain_file=None, train_mode=False):
        if domain_type == "DISCRETE" and domain_file == None:
            print("Creation of Negotiation need to specify domain_type and domain_file.")
            exit(-1)
        self.max_round = max_round
        self.issue_num = issue_num
        self.render = render
        self.agents_pool = []
        self.score_table = {}
        self.round_table = []
        self.total_score_table = {}
        self.total_round_table = []
        self.domain_type = domain_type
        self.domain_file = domain_file
        self.train_mode = train_mode

        if domain_type == "DISCRETE":

            self.issue_num = 0
            self.issue_name = []

            self.agentA_discount = 1
            self.agentB_discount = 1  # discount
            self.agentA_reservation = 0.0  
            self.agentB_reservation = 0.0

            self.agentA_issue_value = []  
            self.agentB_issue_value = []
            self.agentA_issue_weight = []
            self.agentB_issue_weight = []
            self.agentA_best_bid = []
            self.agentB_best_bid = []
            self.bidSpace = 1  


   
    def init(self):

        self.issue_num = 0
        self.issue_name = []

        self.agentA_discount = 1
        self.agentB_discount = 1  
        self.agentA_reservation = 0.0  
        self.agentB_reservation = 0.0

        self.agentA_issue_value = []  
        self.agentB_issue_value = []
        self.agentA_issue_weight = []
        self.agentB_issue_weight = []
        self.agentA_best_bid = []
        self.agentB_best_bid = []
        self.bidSpace = 1 


    def add(self, agent):
        self.agents_pool.append(agent)
        self.score_table[agent.name] = []
        self.total_score_table[agent.name] = []

    def clear(self):
        self.agents_pool = []
        self.score_table = {}
        self.round_table = []
        self.total_score_table = {}
        self.total_round_table = []

    def parseXML(self, domain_file):
        discrete_domain_name = domain_file  # "Acquisition"

        # --domain
        #    |--Acquisition
        #         |--Acquisition_util1.xml
        #         |--Acquisition_util2.xml 
        system_path="/home/ubuntu/multi_issue_negotiation/"
        filepath_1 = system_path+"domain/" + discrete_domain_name + "/" + discrete_domain_name + "_util1.xml"
        filepath_2 = system_path+"domain/" + discrete_domain_name + "/" + discrete_domain_name + "_util2.xml"

        DOMTree_1 = xml.dom.minidom.parse(filepath_1)
        DOMTree_2 = xml.dom.minidom.parse(filepath_2)

        """ for agent A """
        collection_1 = DOMTree_1.documentElement  # utility_space?

        #  get discount_factor
        discount_factor_1 = collection_1.getElementsByTagName("discount_factor")
        
        if discount_factor_1.length == 0:  
            pass
        elif discount_factor_1[0].hasAttribute("value"):
            self.agentA_discount = float(discount_factor_1[0].getAttribute("value"))

        #  get reservation
        reservation_1 = collection_1.getElementsByTagName("reservation")
        if reservation_1.length == 0:
            pass
        elif reservation_1[0].hasAttribute("value"):
            self.agentA_reservation = float(reservation_1[0].getAttribute("value"))
        
        # get  objective
        objective_1 = collection_1.getElementsByTagName("objective")[0]
        issues_1 = objective_1.getElementsByTagName("issue")
        
        # get issue values
        for issue in issues_1:
            self.issue_num += 1
            self.issue_name.append(issue.getAttribute("name"))

            items_1 = issue.getElementsByTagName("item")

            max_evaluation = -9999999
            dict_item = {}
            count = 0  
            for item in items_1:
                count += 1
                int_eval = float(item.getAttribute("evaluation"))
                dict_item[item.getAttribute("value")] = int_eval  # { "1 million $": 8 , "2.5 million $": 8, ...}
                
                if max_evaluation < int_eval:
                    max_evaluation = int_eval
            
            # normalization
            max_key = None
            for key in dict_item.keys():
                if dict_item[key] == max_evaluation:
                    if max_key is None:
                        max_key = key
                dict_item[key] = dict_item[key] / max_evaluation

            # bid space size
            self.bidSpace *= count
            # best bid
            self.agentA_best_bid.append(max_key)
            self.agentA_issue_value.append(dict_item)
        
        # get weights
        weights_1 = objective_1.getElementsByTagName("weight")
        for weight in weights_1:
            self.agentA_issue_weight.append(float(weight.getAttribute("value")))
        
        """ for agent B """
        collection_2 = DOMTree_2.documentElement  # utility_space

        #  get discount_factor
        discount_factor_2 = collection_2.getElementsByTagName("discount_factor")
        if discount_factor_2.length == 0:
            pass
        elif discount_factor_2[0].hasAttribute("value"):
            self.agentB_discount = float(discount_factor_2[0].getAttribute("value"))

        #  reservation
        reservation_2 = collection_2.getElementsByTagName("reservation")
        if reservation_2.length == 0:
            pass
        elif reservation_2[0].hasAttribute("value"):
            self.agentB_reservation =float(reservation_2[0].getAttribute("value"))
        
        # objective
        objective_2 = collection_2.getElementsByTagName("objective")[0]
        issues_2 = objective_2.getElementsByTagName("issue")
        
        # issue values
        for issue in issues_2:
            # self.issue_num += 1
            # self.issue_name.append(issue.getAttribute("name"))

            items_2 = issue.getElementsByTagName("item")

            max_evaluation = -9999999
            dict_item = {}
            # count = 0  
            for item in items_2:
                # count += 1
                int_eval = float(item.getAttribute("evaluation"))
                dict_item[item.getAttribute("value")] = int_eval  # { "1 million $": 8 , "2.5 million $": 8, ...}
                if max_evaluation < int_eval:
                    max_evaluation = int_eval
            
            # normalization
            max_key = None
            for key in dict_item.keys():
                if dict_item[key] == max_evaluation:
                    max_key = key
                dict_item[key] = dict_item[key] / max_evaluation

            
            self.agentB_best_bid.append(max_key)
            self.agentB_issue_value.append(dict_item)
        
        # weights
        weights_2 = objective_2.getElementsByTagName("weight")
        for weight in weights_2:
            self.agentB_issue_weight.append(float(weight.getAttribute("value")))


    def reset(self, opposition="low"):
        if self.domain_type == "REAL":
            assert len(self.agents_pool) == 2
            if self.train_mode:
                random.shuffle(self.agents_pool)
            prefer_list = self.gen_preferlist(opposition)
            if self.render:
                print("\na new negotiation start!\n")
            for i in range(2):
                self.agents_pool[i].set_prefer(prefer_list[i])
                self.agents_pool[i].set_condition(1 - i)
                self.agents_pool[i].discount = 1.0  
                if self.render:
                    print("%s 's prefer: " % self.agents_pool[i].name, self.agents_pool[i].prefer)
                    print("%s 's condition: " % self.agents_pool[i].name, self.agents_pool[i].condition)
                self.agents_pool[i].reset()
            if self.render:
                print()
        elif self.domain_type == "DISCRETE":
            self.init()
            self.parseXML(self.domain_file)
            assert len(self.agents_pool) == 2
            
            if self.render:
                print(self.domain_file, "Using the issue weight of the original domain, the calculated opposition degree is:", self.calc_opposition([self.agentA_issue_weight, self.agentB_issue_weight]))

            self.agents_pool[0].issue_value = self.agentA_issue_value
            self.agents_pool[1].issue_value = self.agentB_issue_value
            self.agents_pool[0].issue_weight = self.agentA_issue_weight
            self.agents_pool[1].issue_weight = self.agentB_issue_weight
            self.agents_pool[0].prefer = self.agentA_issue_weight
            self.agents_pool[1].prefer = self.agentB_issue_weight
            self.agents_pool[0].bestBid = self.agentA_best_bid
            self.agents_pool[1].bestBid = self.agentB_best_bid
            self.agents_pool[0].discount = self.agentA_discount
            self.agents_pool[1].discount = self.agentB_discount
            self.agents_pool[0].reservation = self.agentA_reservation
            self.agents_pool[1].reservation = self.agentB_reservation
            if self.render:
                print("\na new negotiation start!\n")           
            for i in range(2):
                self.agents_pool[i].set_condition(1 - i)
                self.agents_pool[i].issue_name = self.issue_name
                self.agents_pool[i].bidSpace =  self.bidSpace
                self.agents_pool[i].issue_num = self.issue_num
                self.agents_pool[i].oppo_prefer = [1] * self.issue_num
                self.agents_pool[i].domain_type = self.domain_type
                if self.render:
                    print("%s 's prefer: " % self.agents_pool[i].name, self.agents_pool[i].prefer)
                    print("%s 's condition: " % self.agents_pool[i].name, self.agents_pool[i].condition)
                self.agents_pool[i].reset()
            if self.render:
                print()

    def run(self):  # design for bilateral negotiation baseline agent
        current_player = 0
        last_action = None
        is_accept = False
        for i in range(1, self.max_round + 1):
            if self.render:
                print("Round:", i)
            current_player = 1 - i % 2
            self.agents_pool[current_player].set_t(i)
            self.agents_pool[1-current_player].set_t(i)
            last_action = self.agents_pool[current_player].act()

            if (last_action is None) and (self.agents_pool[current_player].__class__.__name__ == "CUHKAgent" or self.agents_pool[current_player].__class__.__name__ == "HardHeadedAgent" \
                or self.agents_pool[current_player].__class__.__name__ == "YXAgent" or self.agents_pool[current_player].__class__.__name__ == "OMAC" \
                or self.agents_pool[current_player].__class__.__name__ == "AgentLG" or self.agents_pool[current_player].__class__.__name__ == "ParsAgent" \
                or self.agents_pool[current_player].__class__.__name__ == "Caduceus"):
                if self.agents_pool[current_player].accept == True:
                    print("\n", self.agents_pool[current_player].name, "accept.") 
                    is_accept = True  
                    self.score_table[self.agents_pool[current_player].name].append(self.agents_pool[current_player].utility_received[-1])
                    self.score_table[self.agents_pool[1-current_player].name].append(self.agents_pool[1-current_player].utility_proposed[-1])
                    self.round_table.append(i)                    
                elif self.agents_pool[current_player].terminate == True:   
                    print("\n", self.agents_pool[current_player].name, "end the negotiation.") 
                else:
                    print("Something wrong.")
                    exit(0)
                break

            if self.render:
                print("  "+self.agents_pool[current_player].name, "gives an offer:", last_action)
                print("  utility to %s: %f, utility to %s: %f\n" % (self.agents_pool[current_player].name, get_utility(last_action, self.agents_pool[current_player].prefer, self.agents_pool[current_player].condition, self.agents_pool[current_player].domain_type, self.agents_pool[current_player].issue_value), self.agents_pool[1 - current_player].name, get_utility(last_action, self.agents_pool[1 - current_player].prefer, self.agents_pool[1 - current_player].condition, self.agents_pool[1 - current_player].domain_type, self.agents_pool[1 - current_player].issue_value)))
            
            self.agents_pool[1 - current_player].receive(last_action)
            if self.agents_pool[1 - current_player].accept:
                is_accept = True
                if self.render:
                    print("  "+self.agents_pool[1 - current_player].name, "accept the offer.\n")
                self.score_table[self.agents_pool[current_player].name].append(self.agents_pool[current_player].utility_proposed[-1])
                self.score_table[self.agents_pool[1-current_player].name].append(self.agents_pool[1-current_player].utility_received[-1])
                self.round_table.append(i)
                break

        if self.render:
            if not is_accept:
                print("Negotiaion Failed!\n")
            else :
                print("Negotiation Successed!\n")

        if not is_accept:
            self.total_score_table[self.agents_pool[0].name].append(0)
            self.total_score_table[self.agents_pool[1].name].append(0)
            self.total_round_table.append(self.max_round)
        else:
            self.total_score_table[self.agents_pool[0].name].append(self.score_table[self.agents_pool[0].name][-1])
            self.total_score_table[self.agents_pool[1].name].append(self.score_table[self.agents_pool[1].name][-1])
            self.total_round_table.append(self.round_table[-1])


    def gen_preferlist(self, opposition_type):
        opposition = 1
        if opposition_type == "low":
            while opposition > 0.3:
                prefer_list = [self.gen_prefer(), self.gen_prefer()]
                opposition = self.calc_opposition(prefer_list)                
        elif opposition_type == "high":
            while opposition < 0.3 or opposition > 0.5:
                prefer_list = [self.gen_prefer(), self.gen_prefer()]
                opposition = self.calc_opposition(prefer_list)
        elif opposition_type == "mix":
            while opposition > 0.5:
                prefer_list = [self.gen_prefer(), self.gen_prefer()]
                opposition = self.calc_opposition(prefer_list)
        if self.render:
            print("opposition : ", opposition)
        return prefer_list

    def gen_prefer(self):
        prefer = []
        total = 1
        for i in range(self.issue_num):
            if i == self.issue_num - 1:
                prefer.append(total)
            else:
                val = random.uniform(0.1, total - (self.issue_num - (i + 1)) * 0.1)
                prefer.append(val)
                total -= val
        return prefer

    def calc_opposition(self, prefer_list):
        if self.domain_type == "REAL":
            minn = math.hypot(1, 1)
            for i in range(100):
                offer1 = [random.random() for j in range(self.issue_num)]
                offer2 = [1 - item for item in offer1]
                u1 = 0
                u2 = 0
                for i in range(len(offer1)):
                    u1 += offer1[i] * prefer_list[0][i]
                for i in range(len(offer2)):
                    u2 += offer2[i] * prefer_list[1][i]
                minn = min(math.hypot(1 - u1, 1 - u2), minn)
            return minn
        elif self.domain_type == "DISCRETE": 
            minn = math.hypot(1.0, 1.0)
            _allBids = self.getAllBids()               
            for _bid in _allBids:
                # u1 = get_utility(_bid, self.agentA_issue_weight, 1, self.domain_type, self.agentA_issue_value)
                # u2 = get_utility(_bid, self.agentB_issue_weight, 0, self.domain_type, self.agentB_issue_value)
                u1 = get_utility(_bid, prefer_list[0], 1, self.domain_type, self.agentA_issue_value)
                u2 = get_utility(_bid, prefer_list[1], 0, self.domain_type, self.agentB_issue_value)
                u3 = math.hypot(1.0-u1, 1.0-u2)
                if u3 < minn:
                    minn = u3
            return minn 


    def getAllBids(self):
        _allBids = []
        issueValues = self.getIssueValues(issue_value_type = self.domain_type)
        resList = self.lists_combination(issueValues,'@')
        for i in range(len(resList)):
            tmpList = resList[i].split('@')
            _allBids.append(tmpList)

        # firstValueBid = []
        # for i in range(self.issue_num):
        #     firstValueBid.append(issueValues[i][0])
        # _allBids.append(firstValueBid)

        # for i in range(self.issue_num):
        #     tmpBids = []

        #     for bid in _allBids: # [[0,0,0]]
        #         tmpBid = bid
        #         for value in issueValues[i]:                  
        #             tmpBid[i] = value
        #             tmpBids.append(copy.deepcopy(tmpBid))
            
        #     _allBids = tmpBids

        return _allBids

    
    def lists_combination(self, lists, code=''):
       
            
        def myfunc(list1, list2):
            return [str(i)+code+str(j) for i in list1 for j in list2]

        return reduce(myfunc, lists)

    def getIssueValues(self, issue_value_type="REAL"):
        retvals = []
        if issue_value_type == "DISCRETE":
            for i in range(self.issue_num):
                retvals.append(list(self.agentA_issue_value[i].keys()))
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
        