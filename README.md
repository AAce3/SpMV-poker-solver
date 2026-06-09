## Overview

This is an experimental project to analyze the viability of implementing an SpMV-based poker solver, based on the work [GPU-Accelerated Counterfactual Regret Minimization](https://arxiv.org/pdf/2408.14778) by Prof. Juho Kim at the University of Toronto. 

### CFR Algorithm:

Traditional poker solvers use a technique called CFR (Counterfactual Regret) to iteratively refine a strategy (action probability for a certain state). CFR will iterate on a strategy until it finds an equilibrium, in which the EV of each action is identical. 

Suppose on the river, we have the following game tree:

```
Root
├── Hero check
│   ├── Villain check
│   │   └── Showdown
│   └── Villain bet
│       ├── Hero fold
│       │   └── Hero loses pot
│       └── Hero call
│           └── Showdown (EV is determined by ranges)
└── Hero bet
    ├── Villain fold
    │   └── Hero wins pot
    └── Villain call
        └── Showdown (EV is determined by ranges)
```

Suppose that Hero's initial strategy is to check 50% of the time and bet 50% of the time. 

The CFR algorithm will then take the following steps:
First, it will compute the EV of the entire 'check' subtree and the entire 'bet' subtree based on the fixed strategies of the Hero and the Villain. In other words, we compute $\sum_{\text{all trajectories } t} V(t)p(t)$ across all trajectories (e.g. Hero Bet/Villain Call, or Hero Check/Villain Check), where $p$ is the probability of taking that trajectory and $V$ is the value of said trajectory. 

Suppose in this case that the EV of bet is 7, and the EV of check is 3. Thus, the EV of betting is higher than the EV of checking, meaning that the algorithm will adjust their strategy to bet more often in the next iteration. 

##### Regret Matching

To determine how much more often to bet, CFR first converts the action EVs into regret deltas by comparing each action against the EV of the current strategy

CFR keeps a running regret value $R(a)$ for each action. After each iteration, the new regret delta is added to this accumulated value $R(a) \leftarrow R(a) + \Delta R(a)$

From the example above, we have $\Delta R(\text{check}) = -2, \space \Delta R(\text{bet}) = 2$. If the previous accumulated regrets were $R(\text{check}) = 5,  \space R(\text{bet}) = 5$, then after this update, we have $R(\text{check}) = 5 - 2 = 3$ and $R(\text{bet}) = 5 + 2 = 7$

To get the next strategy, CFR ignores negative regrets and normalizes the positive ones:

$p(a) = \frac{\max(R(a), 0)}{\sum_{a'} \max(R(a'), 0)}$

So,  we have $p(\text{check}) = \frac{3}{3+7} = 0.3$ and $p(\text{bet}) = \frac{7}{3+7} = 0.7$

Thus, the next strategy checks 30% of the time and bets 70% of the time.

##### Recursive Strategy Updates

Of course, the villain's strategy does not remain constant either. Suppose we zoom into the "Hero Check" node:

```
Hero check
├── Villain check
│   └── Showdown
└── Villain bet
    ├── Hero fold
    │   └── Hero loses pot
    └── Hero call
        └── Showdown (EV is determined by ranges)
```

CFR will do the same thing for villain's decision. It will compute the EV of Villain checking, and the EV of villain betting, and update its strategy accordingly.


##### Information Sets

In poker, CFR does not update strategies for fully-known game states where both players' hands are visible. Instead, it updates strategies for information sets, which represent what the player actually knows when making a decision.

For example, when Hero faces a river bet, Hero knows their own hand, the board, and the betting history, but not Villain's exact hand. So an information set can be thought of as: $I = (\text{Hero hand}, \text{board}, \text{betting history})$

All possible Villain hands that could have reached this spot are grouped into the same information set from Hero's perspective. Therefore, when CFR computes $\text{EV}(\text{call})$ or $\text{EV}(\text{fold})$, it is not evaluating against one specific Villain hand. It is averaging over Villain's possible hands, weighted by how likely each hand is to reach this spot.

This changes the regret update slightly. Instead of updating regret using a single action EV, CFR uses the expected action EV over all hidden states inside the information set: 

$\Delta R(I,a) = \text{EV}(I,a) - \text{EV}(I)$,

where $\text{EV}(I,a)$ already includes the weighted average over possible Villain hands. In other words, the same regret matching process applies, but the EVs are computed over an information set rather than a fully-known game state. 

To compute this weighted average, CFR needs reach probabilities. A reach probability tells us how likely a particular hidden state is under the current strategy. For example, if Villain is very likely to reach this river spot with one hand but almost never reaches it with another, the first hand should contribute much more to Hero's action EV. Thus, reach probabilities determine the weights used when averaging over the hidden states inside an information set.

### Converting to Matrix Multiplications


The main computational workload of CFR consists essentially of two phases.
1. Downward pass to compute probability of each node
2. Upwards propagation to compute EV and regret


For demonstration, imagine a toy game tree:
```
        root
       /    \
      A      B
     / \    / \
    T1 T2  T3 T4
```

And our strategy is the following:

| Action   | Probability |
| -------- | ----------- |
| Root → A | 0.6         |
| Root → B | 0.4         |
| A → T1   | 0.2         |
| A → T2   | 0.8         |
| B → T3   | 0.5         |
| B → T4   | 0.5         |

We can write this in terms of a transition matrix. Number the nodes 0 = root, 1 = A, 2 = B, 3 = T1...

The tree can be represented as a sparse transition matrix $M$, where each nonzero entry corresponds to an edge in the tree and stores the probability of taking that edge:

$$
M =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0.6 & 0 & 0 & 0 & 0 & 0 & 0 \\
0.4 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0.2 & 0 & 0 & 0 & 0 & 0 \\
0 & 0.8 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.5 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.5 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

The first use of Matrix-Vector multiplication is the upward EV pass. Once terminal utilities are known, we can propagate values back toward the root using the transpose of this matrix with the formula $v_{\text{parent}} = M^T v_{\text{child}}$.

To visualize how this works, suppose the terminal values are

| Terminal | Value |
| -------- | ----- |
| T1       | 10    |
| T2       | -5    |
| T3       | 2     |
| T4       | -8    |

Normally, to compute the values of A and B, we would have $v(A) = 0.2 \cdot v(T_1) + 0.8 \cdot v(T_2) = -2$, and similarly $v(B) = 0.5\cdot v(T_3) + 0.5\cdot v(T_4) = -3$. However, we can reformat this in terms of a matrix-vector multiplication. 

Represent the values as a vector over all nodes:

$$v =
\begin{bmatrix}
0 \\
0 \\
0 \\
10 \\
-5 \\
2 \\
-8
\end{bmatrix}
$$

To propagate EV upward, we multiply by the transpose of our transition matrix:

$$
M^T v =
\begin{bmatrix}
0 & 0.6 & 0.4 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0.2 & 0.8 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0.5 & 0.5 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
0 \\
0 \\
10 \\
-5 \\
2 \\
-8
\end{bmatrix}
$$


This gives us

$$
M^T v =
\begin{bmatrix}
0 \\
-2 \\
-3 \\
0 \\
0 \\
0
\end{bmatrix}
$$

So one MV multiplication gives us the EVs of the immediate parent nodes.
Then we apply the same idea again to push values from $A$ and $B$ back to the root.

$$
v =
\begin{bmatrix}
0 \\
-2 \\
-3 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
$$

$$
M^T v =
\begin{bmatrix}
-2.4 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
$$

In other words, the upward EV pass is just repeated sparse matrix-vector multiplication with $M^T$.

We can use these EVs to compute regret values. 
However, we also need reach probability, i.e. how likely each node is under the current strategy, in order to correctly apply the weighted regret update. We compute them with the same sparse matrix vector multiplication, but in the downward direction, using the formula $r_{\text{child}} = M r_{\text{parent}}$.

Starting from the root,


$$
r =
\begin{bmatrix}
1 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
$$

we can compute

$$
Mr =
\begin{bmatrix}
0 \\
0.6 \\
0.4 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
$$

So $A$ is reached with probability $0.6$, and $B$ is reached with probability $0.4$.

Applying the matrix again:


$$
M
\begin{bmatrix}
0 \\
0.6 \\
0.4 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}
= \begin{bmatrix}
0 \\
0 \\
0 \\
0.12 \\
0.48 \\
0.20 \\
0.20
\end{bmatrix}
$$


This gives the reach probabilities of the terminal nodes. In the full CFR update, these reach probabilities are used to weight the regret updates.


So, each individual CFR iteration becomes
1. Use the current strategy to fill the transition matrix $M$
2. Use $M^T v$ to propagate EV upward from terminal nodes
3. Use $Mr$ to propagate reach probabilities downward from the root
4. Use the EV and reach probabilities to update regrets
5. Normalize positive regrets to get the next strategy

### Why SpMV?

SpMV stands for Sparse Matrix-Vector Multiplication. It refers to multiplying a matrix by a vector when most entries in the matrix are zero.

A game tree is naturally sparse. Each node only connects to a few children, even though the full matrix contains an entry for every possible pair of nodes. For example, in the toy tree above, the transition matrix is mostly zeros, and only the actual tree edges contain useful values. Instead of storing the full matrix, SpMV stores only the nonzero entries. Then matrix-vector multiplication becomes a pass over these edges. 

The advantage is that this avoids recursive tree traversal. Instead of walking the game tree node by node, we can process many edges in parallel. This is especially attractive on GPUs, which are designed for large batches of simple numeric operations. And, many highly optimized SpMV routines exist already, for both GPU and for CPU. However, this approach requires more memory.


## Roadmap:

1. Build a river-only CPU baseline solver using CFR+ to give us a correct reference solver
2. Convert the solver to a matrix formulation using the algorithms detailed above
3. Scale to turn and flop subgames
4. Move to GPU using CUDA
