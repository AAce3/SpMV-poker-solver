#include "spmv_poker/game_tree.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace spmv_poker {

NodeIndex GameTree::add_fold_node(float payoff) {
  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  nodes.push_back(GameNode{.type = NodeType::Fold, .payoff = payoff});
  return index;
}

NodeIndex GameTree::add_showdown_node(uint32_t runout_index, float win_payoff,
                                      float loss_payoff) {
  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  nodes.push_back(GameNode{.type = NodeType::Showdown,
                           .runout_index = runout_index,
                           .win_payoff = win_payoff,
                           .loss_payoff = loss_payoff});
  return index;
}

NodeIndex GameTree::add_chance_node(std::span<const GameEdge> children) {
  assert(!children.empty());
  float probability_sum = 0.0F;
  for (const GameEdge &edge : children) {
    assert(edge.child < nodes.size());
    assert(edge.probability >= 0.0F);
    probability_sum += edge.probability;
  }
  assert(std::abs(probability_sum - 1.0F) < 0.00001F);

  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  uint32_t edge_begin = static_cast<uint32_t>(edges.size());
  edges.insert(edges.end(), children.begin(), children.end());
  nodes.push_back(GameNode{.type = NodeType::Chance,
                           .edge_range = {edge_begin, children.size()}});
  return index;
}

NodeIndex GameTree::add_decision_node(Player player,
                                      std::span<const NodeIndex> children) {
  assert(!children.empty());
  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  uint32_t edge_begin = static_cast<uint32_t>(edges.size());
  for (NodeIndex child : children) {
    assert(child < nodes.size());
    edges.push_back(GameEdge{.child = child, .probability = 0.0F});
  }

  uint32_t decision_index = static_cast<uint32_t>(decisions.size());
  size_t offset = state.regrets.size();
  size_t decision_hand_count = hand_count_for(player);
  size_t entry_count = decision_hand_count * children.size();
  state.regrets.resize(offset + entry_count, 0.0F);
  state.strategy_sum.resize(offset + entry_count, 0.0F);
  decisions.push_back(DecisionNode{
      .entries = {offset, entry_count},
      .hand_count = decision_hand_count,
      .action_count = children.size(),
  });

  nodes.push_back(
      GameNode{.type = NodeType::Decision,
               .edge_range = {edge_begin, children.size()},
               .decision_index = decision_index,
               .player = player});
  return index;
}

} // namespace spmv_poker
