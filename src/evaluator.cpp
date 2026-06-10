#include "spmv_poker/evaluator.h"

#include <algorithm>
#include <cassert>

namespace spmv_poker {

std::span<float> RecursiveEvaluator::values(size_t depth) {
  return IndexRange{depth * value_stride, value_stride}.view(
      value_scratch);
}

std::span<float> RecursiveEvaluator::reach(size_t depth) {
  return IndexRange{depth * reach_stride, reach_stride}.view(
      reach_scratch);
}

std::span<float> RecursiveEvaluator::strategy(size_t depth,
                                              const DecisionNode &node) {
  return IndexRange{depth * strategy_stride, node.entry_count()}.view(
      strategy_scratch);
}

void RecursiveEvaluator::evaluate(Player player, const Range &opponent_range,
                                  std::vector<float> &output_values) {
  assert(tree.root < tree.nodes.size());
  Player opponent_player = opponent(player);
  assert(opponent_range.hands == terminals.hand_tables[static_cast<size_t>(
                                      opponent_player)]);
  assert(opponent_range.weights.size() == tree.hand_count_for(opponent_player));
  assert(terminals.hands(player).size() == tree.hand_count_for(player));
  size_t depth_count = maximum_depth(tree.root) + 1;
  value_stride = tree.hand_count_for(player);
  reach_stride = tree.hand_count_for(opponent_player);
  value_scratch.resize(depth_count * value_stride);
  reach_scratch.resize(depth_count * reach_stride);
  strategy_stride = 0;
  for (const DecisionNode &decision : tree.decisions) {
    strategy_stride = std::max(strategy_stride, decision.entry_count());
  }
  strategy_scratch.resize(depth_count * strategy_stride);
  std::copy(opponent_range.weights.begin(), opponent_range.weights.end(),
            reach(0).begin());

  evaluate_node(tree.root, player, 0);
  auto root_values = values(0);
  output_values.assign(root_values.begin(), root_values.end());
}

void RecursiveEvaluator::evaluate_node(NodeIndex node_index, Player player,
                                       size_t depth) {
  const GameNode &node = tree.nodes[node_index];
  auto opponent_range = reach(depth);
  auto node_values = values(depth);

  switch (node.type) {
  case NodeType::Fold:
    terminals.apply_fold(player, opponent_range,
                         player == Player::Hero ? node.payoff : -node.payoff,
                         node_values);
    return;

  case NodeType::Showdown:
    if (player == Player::Hero) {
      terminals.apply_showdown(player, node.runout_index, opponent_range,
                               node.win_payoff, node.loss_payoff, node_values);
    } else {
      terminals.apply_showdown(player, node.runout_index, opponent_range,
                               -node.loss_payoff, -node.win_payoff,
                               node_values);
    }
    return;

  case NodeType::Chance: {
    std::fill(node_values.begin(), node_values.end(), 0.0F);
    for (const GameEdge &edge : tree.children(node)) {
      std::copy(opponent_range.begin(), opponent_range.end(),
                reach(depth + 1).begin());
      evaluate_node(edge.child, player, depth + 1);
      auto child_values = values(depth + 1);
      for (size_t hand = 0; hand < value_stride; ++hand) {
        node_values[hand] += edge.probability * child_values[hand];
      }
    }
    return;
  }

  case NodeType::Decision: {
    const DecisionNode &decision = tree.decisions[node.decision_index];
    auto strategy = this->strategy(depth, decision);
    tree.state.strategy(decision, strategy);
    auto children = tree.children(node);
    std::fill(node_values.begin(), node_values.end(), 0.0F);

    for (size_t action = 0; action < children.size(); ++action) {
      NodeIndex child = children[action].child;
      auto child_range = reach(depth + 1);
      if (node.player == player) {
        std::copy(opponent_range.begin(), opponent_range.end(),
                  child_range.begin());
      } else {
        for (size_t hand = 0; hand < reach_stride; ++hand) {
          child_range[hand] =
              opponent_range[hand] *
              strategy[hand * decision.action_count + action];
        }
      }

      evaluate_node(child, player, depth + 1);
      auto child_values = values(depth + 1);
      for (size_t hand = 0; hand < value_stride; ++hand) {
        float action_weight =
            node.player == player
                ? strategy[hand * decision.action_count + action]
                : 1.0F;
        node_values[hand] += action_weight * child_values[hand];
      }
    }
    return;
  }
  }
}

size_t RecursiveEvaluator::maximum_depth(NodeIndex node_index) const {
  size_t depth = 0;
  for (const GameEdge &edge : tree.children(tree.nodes[node_index])) {
    depth = std::max(depth, maximum_depth(edge.child) + 1);
  }
  return depth;
}

} // namespace spmv_poker
