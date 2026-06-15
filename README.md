## Overview

This project is an experimental poker solver built around CFR (Counterfactual Regret Minimization). The goal is to keep the solver structure simple enough to reason about, but organized in a way that can later scale to larger games and faster backends.

The key idea in CFR is that each player repeatedly updates their strategy based on the EV of each action. Over time, positive and negative regrets push the strategy toward equilibrium.

### CFR Algorithm

Traditional poker solvers use CFR to iteratively refine strategy. A strategy is the action probability for a given state, and CFR keeps adjusting those probabilities until the EV of each action becomes equal.

Suppose on the river, we have the following game tree:

```text
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

Suppose Hero starts by checking 50% of the time and betting 50% of the time.

The first CFR step is to evaluate the EV of the full `check` subtree and the full `bet` subtree using the current strategies. If betting has EV 7 and checking has EV 3, then the algorithm increases the probability of betting on the next iteration.

#### Regret Matching

To decide how much to change the strategy, CFR converts action EVs into regrets.

The accumulated regret for an action is updated each iteration:

$$
R(a) \leftarrow R(a) + \Delta R(a)
$$

If checking is below the current strategy EV and betting is above it, the regrets move in opposite directions. CFR then uses only the positive regrets and normalizes them:

$$
p(a) = \frac{\max(R(a), 0)}{\sum_{a'} \max(R(a'), 0)}
$$

That normalized value becomes the next strategy.

#### Recursive Strategy Updates

The same process applies at every decision node. If we zoom into the `Hero check` branch, Villain’s strategy is also updated from the EVs of Villain’s available actions.

#### Information Sets

CFR does not update a strategy for fully observed game states. It updates information sets: what a player knows when making a decision.

For example, when Hero faces a river bet, Hero knows:

- their own hand
- the board
- the betting history

but not Villain’s exact hand.

So an information set is really:

$$
I = (\text{Hero hand}, \text{board}, \text{betting history})
$$

The EV of each action is then averaged over all hidden opponent hands that could have reached the spot, weighted by their reach probabilities.

## Architecture

The main idea is that there is a lot of shared structure inside a street, and
we take advantage of that by compiling the street once into an execution plan
that is reused across boards and across iterations.

For a given street, the betting structure does not change from board to board.
The same actions exist, the same decision nodes exist, and the same traversal
order exists. What changes is the concrete board state and the hand values
attached to that board. The solver separates those two concerns: the street is
compiled once into a fixed execution plan, and each board carries its own
regrets, strategies, and reach values. That means the hot path is not rebuilding
tree structure. It is reusing the same compiled plan across many boards and many
iterations.

The board dimension is treated as a batch dimension. The evaluator can process
multiple boards in one pass while using the same compiled street plan. That is
useful because the traversal order is shared, the action layout is shared, the
decision nodes are shared, and only the board-specific values differ. This
shared structure is the main reason the current implementation is organized the
way it is.

The evaluator also uses reusable scratch memory for reach propagation, value
propagation, and street continuation. The goal is to avoid allocating temporary
vectors inside the hot path. The solver does more planning up front so repeated
iterations can reuse the same buffers.

Terminal evaluation is separated from tree traversal. The evaluator asks the
terminal layer for fold values, showdown values, and board-specific masking.
This keeps terminal logic separate from traversal logic, which makes both parts
easier to change independently.

Hero and Villain use separate hand tables. That avoids forcing both players into
the same hand universe when their ranges differ. The payoff is cleaner solver
state and less wasted work on hands that are absent from a player’s range.

Some streets end in a boundary rather than a terminal. In that case, the solver
can continue into the next street using child jobs built from the current board
state. This allows turn-root and river-root solves to share the same overall
solver structure.


## Roadmap:

1. [x] Build a river-only CPU baseline solver using CFR+ to give us a correct reference solver
2. [x] Convert the solver to a matrix formulation using the algorithms detailed above
3. [x] Scale to turn and flop subgames
4. Move to GPU using CUDA
