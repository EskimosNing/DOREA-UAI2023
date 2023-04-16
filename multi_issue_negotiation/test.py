from atlas3 import Atlas3
from agentlg import AgentLG
from ponpokoagent import PonPokoAgent
#from parscat import ParsCat
from omacagent import OMAC
from CUHKAgent import CUHKAgent
from HardHeaded import HardHeadedAgent
from YXAgent import YXAgent
from ParsAgent import ParsAgent
#from caduceus import Caduceus
from utils import *
from agent import *
from negotiation import *
import numpy as np
import pandas as pd


if __name__ == "__main__": 
    ponpoko = PonPokoAgent(max_round=30, name="ponpoko agent")
    #parscat = ParsCat(max_round=30, name="parscat agent")
    atlas3 = Atlas3(max_round=30, name="atlas3 agent")
    omac = OMAC(max_round=30, name="omac agent")
    agentlg = AgentLG(max_round=30, name="AgentLG agent")
    cuhkagent = CUHKAgent(max_round=30, name="CUHK agent")
    hardheaded = HardHeadedAgent(max_round=30, name="HardHeaded agent")
    yxagent = YXAgent(max_round=30, name="YXAgent")
    parsagent = ParsAgent(max_round=30, name="ParsAgent")
    #caduceus = Caduceus(max_round=30, name="Caduceus agent")


    allDomains = ["Acquisition", "Animal", "Coffee", "DefensiveCharms", "Camera", "DogChoosing", "FiftyFifty2013", "HouseKeeping", "Icecream", "Kitchen", "Laptop", "NiceOrDie", "Outfit", "planes", "SmartPhone", "Ultimatum", "Wholesaler", "RentalHouse-B", "Barter-C", "Amsterdam-B"]


    for i in range(10): 
        negotiation = Negotiation(max_round=30, issue_num=3, render=True, domain_type="DISCRETE", domain_file="Laptop")
        negotiation.add(ponpoko)
        negotiation.add(yxagent)
        negotiation.reset()
        print("outcome space of current domain : ", negotiation.bidSpace)

        negotiation.run()
