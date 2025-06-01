# EX_02 - POLICY EVALUATION

## AIM:

To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT:

The FrozenLake problem is a classic reinforcement learning task in which an agent must learn to navigate a frozen, slippery surface to reach a goal while avoiding holes that terminate the episode. The environment, implemented using OpenAI Gym's FrozenLake-v1, is a grid-based Markov Decision Process (MDP) with stochastic transitions due to the slippery surface.

## Environment Description:

The lake is represented as a grid (default is 4x4), where each cell is a state. The agent starts at the Start (S) state and must reach the Goal (G).

Some tiles are safe (F: Frozen), while others are holes (H). If the agent steps into a hole, the episode ends immediately.

The goal is to reach G while avoiding H.

### State Types:

S: Start state

F: Frozen tile (safe)

H: Hole (danger, episode ends)

G: Goal (success, episode ends)

### Actions:

The agent can move in four directions:

0 – Left

1 – Down

2 – Right

3 – Up

### Transition Probabilities:

Due to the slippery nature of the lake:

The agent may not move in the intended direction.

Instead, the movement follows stochastic transition dynamics, where the agent slips and may move in an unintended direction with a certain probability.

### Rewards:

A reward of +1 is given only when the agent reaches the goal state (G).

All other transitions yield a reward of 0, including falling into a hole or moving on frozen tiles.

## POLICY EVALUATION FUNCTION

```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

V1 = policy_evaluation(pi_frozenlake, P, gamma=0.99)
V2 = policy_evaluation(pi_2, P, gamma=0.99)

print_state_value_function(V1, P, n_cols=4, prec=5)
print_state_value_function(V2, P, n_cols=4, prec=5)


print("Name         : Niraunjana Gayathri G R")
print("Register N0. : 212222230096") 

if(np.sum(V1 >= V2) == 11):
    print("The first policy is the better policy")
elif(np.sum(V2 >= V1) == 11):
    print("The second policy is the better policy")
else:
    print("Both policies have their merits.")

```

## OUTPUT:

![image](https://github.com/user-attachments/assets/0bc31eb1-825f-4071-bbeb-a27580cbec16)

![image](https://github.com/user-attachments/assets/66aa6b96-3dbc-47a8-a35e-2e53c6e94459)

![image](https://github.com/user-attachments/assets/7725fa1d-91f8-4f18-8906-d681fbd50ffa)

![image](https://github.com/user-attachments/assets/ba1606b9-e35b-4de5-968c-1ac0f49eec9e)

![image](https://github.com/user-attachments/assets/c08ce3a8-bf54-468e-b456-a14658f55190)





