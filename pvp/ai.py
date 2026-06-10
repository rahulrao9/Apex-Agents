# this file is for the PVP AI, which will be used in the PVP competition. 
# It will load the model and use it to infer actions based on the observations.
import os
import torch
from neurips2022nmmo import Team

from .agent import NMMOAgent
from .translator import Translator


class PvPAI(Team):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract the team_id assigned by run.py
        team_id = args[0] if len(args) > 0 else kwargs.get('team_id', "RealikunTeam-0")
        
        # Pass the team_id into our agent
        self.agent = NMMOAgent(False, team_id=team_id)
        
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'model.pth')
        agent_dict = torch.load(model_path)
        self.agent.loads(agent_dict)

        self.translator = Translator()
        self.step = 0

    def reset(self):
        self.step = 0

    def act(self, observations):
        if self.step == 0:
            self.translator.reset(observations)
            
        state = self.translator.trans_obs(observations)
        
        # Pass raw obs safely around the tensorizer
        actions = self.agent.infer(state, raw_obs=observations)
        
        actions = self.translator.trans_action(actions)
        self.step += 1
        return actions