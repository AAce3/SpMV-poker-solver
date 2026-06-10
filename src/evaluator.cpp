#include "spmv_poker/evaluator.h"

#include <algorithm>
#include <cassert>

namespace spmv_poker {

void Evaluator::evaluate(Player player, const Range &opponent_range,
                         std::vector<float> &values) const {
  assert(tree.hand_count == terminals.hand_table.size());
  assert(opponent_range.weights.size() == tree.hand_count);
  assert(tree.root < tree.nodes.size());
  evaluate_node(tree.root, player, opponent_range, values);
}

void Evaluator::evaluate_node(NodeIndex node_index, Player player,
                              const Range &opponent_range,
                              std::vector<float> &values) const {
  const GameNode &node = tree.nodes[node_index];
  switch (node.type) {
  case NodeType::Fold:
    terminals.apply_fold(opponent_range,
                         player == Player::Hero ? node.payoff : -node.payoff,
                         values);
    return;

  case NodeType::Showdown:
    if (player == Player::Hero) {
      terminals.apply_showdown(node.runout_index, opponent_range,
                               node.win_payoff, node.loss_payoff, values);
    } else {
      terminals.apply_showdown(node.runout_index, opponent_range,
                               -node.loss_payoff, -node.win_payoff, values);
    }
    return;

  case NodeType::Chance: {
    values.assign(tree.hand_count, 0.0F);
    std::vector<float> child_values;
    for (const GameEdge &edge : tree.children(node)) {
      evaluate_node(edge.child, player, opponent_range, child_values);
      for (size_t hand = 0; hand < tree.hand_count; ++hand) {
        values[hand] += edge.probability * child_values[hand];
      }
    }
    return;
  }

  case NodeType::Decision: {
    const DecisionNode &decision = tree.decisions[node.decision_index];
    auto strategy = decision.entries.view(tree.state.strategy);
    auto children = tree.children(node);
    values.assign(tree.hand_count, 0.0F);

    if (node.player == player) {
      std::vector<float> child_values;
      for (size_t action = 0; action < children.size(); ++action) {
        evaluate_node(children[action].child, player, opponent_range,
                      child_values);
        for (size_t hand = 0; hand < tree.hand_count; ++hand) {
          values[hand] += strategy[hand * decision.action_count + action] *
                          child_values[hand];
        }
      }
      return;
    }

    for (size_t action = 0; action < children.size(); ++action) {
      Range action_range;
      action_range.weights.resize(tree.hand_count);
      for (size_t hand = 0; hand < tree.hand_count; ++hand) {
        action_range.weights[hand] =
            opponent_range.weights[hand] *
            strategy[hand * decision.action_count + action];
      }

      std::vector<float> child_values;
      evaluate_node(children[action].child, player, action_range, child_values);
      for (size_t hand = 0; hand < tree.hand_count; ++hand) {
        values[hand] += child_values[hand];
      }
    }
    return;
  }
  }
}

} // namespace spmv_poker
