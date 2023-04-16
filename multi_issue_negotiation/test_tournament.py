from atlas3 import Atlas3
from agentlg import AgentLG
from ponpokoagent import PonPokoAgent
from parscat import ParsCat
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent
import random
from utils import *
from agent import *
from negotiation import *
import numpy as np
import pandas as pd  


if __name__ == "__main__":

    rows = []

    negotiation = Negotiation(max_round=60, issue_num=3, render=False, domain_type="DISCRETE", domain_file="FiftyFifty2013")
    
    
    agent1 = YXAgent(max_round=60, name="YXAgent")
    agent2 = ParsAgent(max_round=60, name="ParsAgent")
    agent3 = PonPokoAgent(max_round=60, name="ponpoko agent")
    agent4 = ParsCat(max_round=60, name="parscat agent")
    agent5 = Atlas3(max_round=60, name="atlas agent")
    agent6 = HardHeadedAgent(max_round=60, name="HardHeaded agent")    
    agent7 = AgentLG(max_round=60, name="AgentLG agent")
    agent8 = CUHKAgent(max_round=60, name="CUHK agent")
    
    agent1_2 = YXAgent(max_round=60, name="YXAgent_2")
    agent2_2 = ParsAgent(max_round=60, name="ParsAgent_2")
    agent3_2 = PonPokoAgent(max_round=60, name="ponpoko agent_2")
    agent4_2 = ParsCat(max_round=60, name="parscat agent_2")
    agent5_2 = Atlas3(max_round=60, name="atlas agent_2")
    agent6_2 = HardHeadedAgent(max_round=60, name="HardHeaded agent_2")    
    agent7_2 = AgentLG(max_round=60, name="AgentLG agent_2")
    agent8_2 = CUHKAgent(max_round=60, name="CUHK agent_2")
    
    agent_group_1 = [agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8]
    agent_group_2 = [agent1_2, agent2_2, agent3_2, agent4_2, agent5_2, agent6_2, agent7_2, agent8_2]

    for i in range(len(agent_group_1)):
        for j in range(len(agent_group_2)):

            row = []
            negotiation.clear()
            negotiation.add(agent_group_1[i])
            negotiation.add(agent_group_2[j])

            row.append(agent_group_1[i].__class__.__name__)  
            row.append(agent_group_2[j].__class__.__name__)

            allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", "Ultimatum", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]
            # allDomains = ["planes"]

            for k in range(100): 
                negotiation.reset(opposition="low", domain_file=allDomains[k%len(allDomains)])      
                # print("outcome space of current domain : ", negotiation.bidSpace)
                negotiation.run()

            for item in negotiation.score_table:
                arr1 = negotiation.score_table[item]
                arr2 = negotiation.round_table
                num_successful = len(arr1)
                                    
                row.append(round(np.mean(arr1), 4))  

            for item in negotiation.total_score_table:
                arr1 = negotiation.total_score_table[item]
                arr2 = negotiation.total_round_table
                row.append(round(np.mean(arr1), 4))  # Overall average utility

            row.append(round(np.mean(negotiation.round_table), 2))  # Average number of rounds successfully negotiated
            row.append(round(np.mean(negotiation.total_round_table), 2))  # The average number of rounds negotiated in total
            row.append(len(negotiation.round_table))  # Number of successful negotiations
            row.append(len(negotiation.total_round_table))  # session 

            rows.append(row)


    df1 = pd.DataFrame(rows, columns=['Agent1', 'Agent2', 'utility1_succ', 'utility2_succ', 'utility1_total', 'utility2_total', 'avg_round_succ', 'avg_round_total', 'num_succ',	'num_total'])
    df1.to_excel("output.xlsx")
