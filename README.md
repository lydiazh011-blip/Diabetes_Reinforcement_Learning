1. Based on the FDA-approved UVA/Padova type 1 diabetes physiological simulator [1] and referring to the clinically significant reinforcement learning environment proposed in [2], a reinforcement learning-driven basic insulin administration control framework was constructed to simulate the blood glucose changes of patients over a multi-day time scale. 

2. To ensure that the research results have clinical relevance, This paper first selects two representative actor-critic reinforcement learning algorithms - Proximal Policy Optimization (PPO) and Soft Actor-Critic, systematically analyzing the performance differences between on-policy and off-policy methods in multi-day blood glucose regulation tasks. 

3. This project further independently implements the Dual-PPO control algorithm proposed in the literature [2], and conducts comparative experiments on different reinforcement learning models in the same simulation environment to verify the feasibility and performance of the PPO and SAC models in the insulin administration control task. 



Reference:

[1] Xie, J. (2021). jxx123/simglucose. [online] GitHub. Available at: https://github.com/jxx123/simglucose.

[2] Marchetti, A., Sasso, D., D’Antoni, F., Morandin, F., Parton, M., Matarrese, M.A.G. and Merone, M. (2025). Deep reinforcement learning for Type 1 Diabetes: Dual PPO controller for personalized insulin management. Computers in Biology and Medicine, 191, p.110147. doi:https://doi.org/10.1016/j.compbiomed.2025.110147.

‌
