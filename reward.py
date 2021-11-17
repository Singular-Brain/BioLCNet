import torch

from bindsnet.learning.reward import AbstractReward



class DynamicDopamineInjection(AbstractReward):
    """
    Dynamic dopamine injection by the dopaminergic layer (the output layer in BioLCNet)
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructor for DynamicDopamineInjection class
        """
        self.reward_predict = torch.tensor(0.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(0.0)  # Predicted reward per episode.
        self.rewards_predict_episode = ([])  # List of predicted rewards per episode (used for plotting).
        self.accumulated_reward = torch.tensor(0.0)
        self.n_labels = kwargs['n_labels']
        self.n_per_class = kwargs['neuron_per_class']

        self.dopamine_base = kwargs['dopamine_base']
        self.rew_base = kwargs['reward_base']
        self.punish_base = kwargs['punishment_base']
        
        self.td_nu = kwargs['td_nu']
        self.dt = torch.as_tensor(kwargs['dt']) 
        self.tc_reward = kwargs['tc_reward']
        self.decay = torch.exp(-self.dt / self.tc_reward)
        self.tc_dps = kwargs.get('tc_dps', 20)
        if self.tc_dps is not None:
            self.decay_dps = torch.exp(-self.dt / self.tc_dps)
        self.dps_factor = kwargs.get('dps_factor', 50)

        self.dps = self.rew_base
        self.neg_dps = self.punish_base
        self.dopamine = self.dopamine_base 
    
    def compute(self, **kwargs) -> None:
        """
        Computes/modifies reward.
        """
        self.dopamine = self.dopamine_base
        self.layers = kwargs['dopaminergic_layers']
        self.label = kwargs['true_label']
        self.give_reward = kwargs['give_reward']
        self.variant = kwargs['variant']
        self.sub_variant = kwargs['sub_variant']

        if self.sub_variant == 'static':
            if self.variant == 'scalar' and self.give_reward:
                if self.label == kwargs['pred_label']:
                    self.dopamine += self.rew_base
                else:
                    self.dopamine += -self.punish_base

            elif self.variant == 'per_spike' and self.give_reward:
                self.dopamine += kwargs['target_spikes'] * self.rew_base - (kwargs['sum_spikes'].sum()-kwargs['target_spikes']) * self.punish_base
            
            elif self.variant == 'per_spike_target' and self.give_reward:
                if self.label == kwargs['pred_label']:
                    self.dopamine += kwargs['pred_spikes'] * self.rew_base
                else:
                    self.dopamine -= kwargs['pred_spikes'] * self.punish_base

        elif self.sub_variant == 'RPE':
            if self.variant == 'scalar':
                if self.give_reward:
                    if self.label == kwargs['pred_label']:
                        self.dopamine += self.rew_base
                    else:
                        self.dopamine -= self.punish_base
                else:
                    self.rew_base = torch.clip(self.rew_base - self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.dps/self.dps_factor,max=self.dps*self.dps_factor)
                    self.punish_base = torch.clip(self.punish_base + self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.neg_dps/self.dps_factor,max=self.neg_dps*self.dps_factor)
            
            elif self.variant == 'per_spike':
                if self.give_reward:
                    self.dopamine += kwargs['target_spikes'] * self.rew_base - (kwargs['sum_spikes'].sum()-kwargs['target_spikes']) * self.punish_base
                else:
                    self.rew_base = torch.clip(self.rew_base - self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.dps/self.dps_factor,max=self.dps*self.dps_factor)
                    self.punish_base = torch.clip(self.punish_base + self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.neg_dps/self.dps_factor,max=self.neg_dps*self.dps_factor)
           
            elif self.variant == 'per_spike_target':
                if self.give_reward:
                    if self.label == kwargs['pred_label']:
                        self.dopamine += kwargs['target_spikes'] * self.rew_base
                    else:
                        self.dopamine -= kwargs['target_spikes'] * self.punish_base
                else:
                    self.rew_base = torch.clip(self.rew_base - self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.dps/self.dps_factor,max=self.dps*self.dps_factor)
                    self.punish_base = torch.clip(self.punish_base + self.td_nu*(self.accumulated_reward-self.reward_predict_episode), min=self.neg_dps/self.dps_factor,max=self.neg_dps*self.dps_factor)
            
            elif self.variant == 'true_pred' or self.variant == 'pure_per_spike':
                self.dps = torch.clip(self.dps - self.td_nu*(self.accumulated_reward-self.reward_predict_episode),min=self.rew_base/self.dps_factor, max=self.rew_base*self.dps_factor)
                self.neg_dps = torch.clip(self.neg_dps + self.td_nu*(self.accumulated_reward-self.reward_predict_episode),min=self.punish_base/self.dps_factor, max=self.punish_base*self.dps_factor)

        elif self.sub_variant == 'pred_decay':
            if self.variant == 'scalar':
                if self.give_reward:
                    if self.label == kwargs['pred_label']:
                        self.dopamine = self.rew_base
                        self.rew_base =  float(torch.clip(torch.tensor([self.rew_base - self.dps/self.dps_factor]),  min=self.dps/self.dps_factor,max=self.dps*self.dps_factor))
                        self.punish_base = float(torch.clip(torch.tensor([self.punish_base + self.neg_dps/self.dps_factor]),  min=self.neg_dps/self.dps_factor,max=self.neg_dps*self.dps_factor))
                    else:
                        self.dopamine = -self.punish_base
                        self.rew_base =  float(torch.clip(torch.tensor([self.rew_base + self.dps/self.dps_factor]),  min=self.dps/self.dps_factor,max=self.dps*self.dps_factor))
                        self.punish_base = float(torch.clip(torch.tensor([self.punish_base - self.neg_dps/self.dps_factor]),  min=self.neg_dps/self.dps_factor,max=self.neg_dps*self.dps_factor))

            elif self.variant == 'true_pred' or self.variant == 'pure_per_spike' or self.variant == 'per_spike' or self.variant == 'per_spike_target':
                assert True, "Not supported"

        else:
            raise ValueError("sub_variant not specified")
        return torch.tensor(self.dopamine)
        
    def update(self, **kwargs) -> None:
        """
        Updates the RPEs and accumulated_reward

        Keyword arguments:

        :param Union[float, torch.Tensor] accumulated_reward: Reward accumulated over
            one episode.
        :param float ema_window: Width of the averaging window.
        """
        # Get keyword arguments.
        self.accumulated_reward = kwargs["accumulated_reward"]
        ema_window = torch.tensor(kwargs.get("ema_window", 10.0))

        # Update RPEs.
        self.reward_predict_episode = (1 - 1 / ema_window) * self.reward_predict_episode + 1 / ema_window * self.accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())

    def online_compute(self, **kwargs) -> None:
        """
        For online rewarding
        """

        if self.label is None:
            return 0.0
        
        s = self.layers.s
        assert s.shape[0] == 1, "This method has not yet been implemented for batch_size>1 !" 
        self.dopamine = (self.decay * (self.dopamine - self.dopamine_base) + self.dopamine_base).to(s.device)
        
        target_spikes = (s[:,self.label*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)                


        if self.variant == "pure_per_spike":
            self.dopamine += target_spikes * self.dps - (s.sum()-target_spikes) * self.neg_dps
        
        elif self.variant == 'true_pred':
            label_spikes = [0.0]*self.n_labels
            for i in range(self.n_labels):
                label_spikes[i] = (s[:,i*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
            if target_spikes == max(label_spikes):
                self.dopamine += target_spikes * self.dps
            else:
                self.dopamine -= max(label_spikes) * self.neg_dps  

        else:
            raise ValueError("variant not specified")
        
        return self.dopamine


class DopaminergicRPE(AbstractReward):
    """
    Dopaminergic RPE class
    """

    def __init__(self, **kwargs) -> None:
        # language=rst
        """
        Constructor for the DopaminergicRPE class
        """
        self.reward_predict = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode = ([])  # List of predicted rewards per episode (used for plotting).\

        self.reward_predict_pos = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode_pos = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode_pos = ([])  # List of predicted rewards per episode (used for plotting).
        
        self.reward_predict_neg = torch.tensor(1.0)  # Predicted reward (per step).
        self.reward_predict_episode_neg = torch.tensor(1.0)  # Predicted reward per episode.
        self.rewards_predict_episode_neg = ([])  # List of predicted rewards per episode (used for plotting).
        
        self.accumulated_reward = torch.tensor(1.0)
        self.variant = None

    def compute(self, **kwargs) -> torch.Tensor:
        """
        Called before each episode
        """

        self.td_nu = kwargs.get('td_nu',0.0001)
        self.dps_base = kwargs.get('dopamine_per_spike_base', 0.01)
        self.negative_dps_base = kwargs.get('negative_dopamine_per_spike_base', 0.0)
        self.layers = kwargs.get('dopaminergic_layers')
        self.n_labels = kwargs.get('n_labels')
        self.n_per_class = kwargs.get('neuron_per_class')
        self.single_output_layer = kwargs['single_output_layer']
        self.tc_reward = kwargs.get('tc_reward')
        self.dopamine_for_correct_pred = kwargs.get('dopamine_for_correct_pred', 1.0)
        self.dopamine_base = kwargs.get('dopamine_base', 0.002)
        dt = torch.as_tensor(self.dt)
        self.decay = torch.exp(-dt / self.tc_reward)
        self.label = kwargs.get('labels', None)
        self.dopamine = self.dopamine_base
        self.variant = kwargs['variant']
        self.sub_variant = kwargs['sub_variant']
        self.dps = self.dps_base
        self.negative_dps = self.negative_dps_base
        
        if self.sub_variant == 'just_decay':
            self.dps = self.dps_base
            self.negative_dps = self.negative_dps_base

        elif self.sub_variant == 'normal':
            if self.accumulated_reward > 0 :
                self.dps = self.dps_base / self.reward_predict_episode_pos
            else :
                self.negative_dps = self.negative_dps_base / self.reward_predict_episode_neg
        
        elif self.sub_variant == 'td_error':
            if self.accumulated_reward > 0 :
                self.dps = self.dps_base - self.td_nu*(self.accumulated_reward-self.reward_predict_episode_pos)
            else :
                self.negative_dps = self.negative_dps_base + self.td_nu*(self.accumulated_reward-self.reward_predict_episode_neg)
        
        else:
            raise ValueError('sub_variant not specified!')


        return self.dopamine


    def update(self, **kwargs) -> None:
        """
        Updates online reward parameters
        """

        # Get keyword arguments.
        self.accumulated_reward = kwargs["accumulated_reward"]
        steps = torch.tensor(kwargs["steps"]).float()
        ema_window = torch.tensor(kwargs.get("ema_window", 10.0))

        # Compute average reward per step.
        self.reward = self.accumulated_reward / steps

        # Update EMAs.
        self.reward_predict = (
            1 - 1 / ema_window
        ) * self.reward_predict + 1 / ema_window * self.reward
        self.reward_predict_episode = (
            1 - 1 / ema_window
        ) * self.reward_predict_episode + 1 / ema_window * self.accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())

        if self.accumulated_reward > 0 :
            self.reward_pos = self.accumulated_reward / steps
            self.reward_predict_pos = (1 - 1 / ema_window) * self.reward_predict_pos + 1 / ema_window * self.reward_pos
            self.reward_predict_episode_pos = (1 - 1 / ema_window) * self.reward_predict_episode_pos + 1 / ema_window * self.accumulated_reward
            self.rewards_predict_episode_pos.append(self.reward_predict_episode_pos.item()) 
        
        else:
            self.reward_neg = self.accumulated_reward / steps
            self.reward_predict_neg = (1 - 1 / ema_window) * self.reward_predict_neg + 1 / ema_window * self.reward_neg
            self.reward_predict_episode_neg = (1 - 1 / ema_window) * self.reward_predict_episode_neg + 1 / ema_window * self.accumulated_reward
            self.rewards_predict_episode_neg.append(self.reward_predict_episode_neg.item())             


    def online_compute(self, **kwargs) -> None:
        """
        Computes online reward
        """

        if self.label is None:
            return 0.0
        
        s = self.layers.s
        self.dopamine = (
                        self.decay
                        * (self.dopamine - self.dopamine_base)
                        + self.dopamine_base
        ).to(s.device)

        target_spikes = (s[:,self.label*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)  

        if self.variant == 'true_pred':
            label_spikes = [0.0]*self.n_labels
            for i in range(self.n_labels):
                label_spikes[i] = (s[:,i*self.n_per_class:(self.label+1)*self.n_per_class,...]).sum().to(s.device)
            if target_spikes == max(label_spikes):
                self.dopamine += target_spikes * self.dps
            else:
                self.dopamine -= max(label_spikes) * self.negative_dps
        elif self.variant == "pure_per_spike":
            self.dopamine += target_spikes * self.dps - (s.sum()-target_spikes) * self.negative_dps
        else:
            raise ValueError("variant not specified")
        
        return self.dopamine
       