#include "spmv_poker/evaluator.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace spmv_poker {

namespace {

void mask_range(std::span<const Hand> hands, uint64_t board_mask,
                std::span<const float> input, std::vector<float> &output) {
  output.resize(input.size());
  for (size_t hand = 0; hand < input.size(); ++hand) {
    output[hand] = (hands[hand].mask & board_mask) == 0 ? input[hand] : 0.0F;
  }
}

void zero_blocked_values(std::span<const Hand> hands, uint64_t board_mask,
                         std::vector<float> &values) {
  for (size_t hand = 0; hand < values.size(); ++hand) {
    if ((hands[hand].mask & board_mask) != 0) {
      values[hand] = 0.0F;
    }
  }
}

} // namespace

void RecursiveEvaluator::evaluate(Player player, const Range &opponent_range,
                                  std::vector<float> &output_values) {
  assert(tree.root < tree.nodes.size());
  Player opponent_player = opponent(player);
  assert(opponent_range.hands == terminals.hand_tables[static_cast<size_t>(
                                      opponent_player)]);
  assert(opponent_range.weights.size() == tree.hand_count_for(opponent_player));
  assert(terminals.hands(player).size() == tree.hand_count_for(player));
  evaluate_node(tree.root, player, opponent_range.weights, output_values, 0);
}

void RecursiveEvaluator::cfr_update(Player player, const Range &player_range,
                                    const Range &opponent_range,
                                    std::vector<float> &output_values,
                                    float iteration_weight) {
  assert(tree.root < tree.nodes.size());
  assert(player_range.hands ==
         terminals.hand_tables[static_cast<size_t>(player)]);
  assert(opponent_range.hands == terminals.hand_tables[static_cast<size_t>(
                                      opponent(player))]);
  assert(player_range.weights.size() == tree.hand_count_for(player));
  assert(opponent_range.weights.size() ==
         tree.hand_count_for(opponent(player)));
  cfr_update_node(tree.root, player, player_range.weights,
                  opponent_range.weights, output_values, 1.0F,
                  iteration_weight, 0);
}

void RecursiveEvaluator::cfr_iteration(const Range &hero_range,
                                       const Range &villain_range,
                                       float iteration_weight) {
  std::vector<float> values;
  cfr_update(Player::Hero, hero_range, villain_range, values,
             iteration_weight);
  cfr_update(Player::Villain, villain_range, hero_range, values,
             iteration_weight);
}

void RecursiveEvaluator::evaluate_node(NodeIndex node_index, Player player,
                                       std::span<const float> opponent_range,
                                       std::vector<float> &node_values,
                                       uint64_t board_mask) {
  const GameNode &node = tree.nodes[node_index];
  node_values.resize(tree.hand_count_for(player));

  switch (node.type) {
  case NodeType::Fold:
    terminal_fold(node, player, opponent_range, node_values, board_mask);
    return;

  case NodeType::Showdown:
    terminal_showdown(node, player, opponent_range, node_values, board_mask);
    return;

  case NodeType::Chance: {
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> child_values;
    for (const GameEdge &edge : tree.children(node)) {
      evaluate_node(edge.child, player, opponent_range, child_values,
                    board_mask | edge.dealt_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        node_values[hand] += edge.probability * child_values[hand];
      }
    }
    return;
  }

  case NodeType::Decision: {
    const DecisionNode &decision = tree.decisions[node.decision_index];
    std::vector<float> strategy(decision.entry_count());
    decision_strategy(decision, false, strategy);
    auto children = tree.children(node);
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> child_values;

    for (size_t action = 0; action < children.size(); ++action) {
      std::span<const float> child_range = opponent_range;
      std::vector<float> filtered_range;
      if (node.player != player) {
        filtered_range.resize(opponent_range.size());
        for (size_t hand = 0; hand < opponent_range.size(); ++hand) {
          bool valid = (terminals.hands(node.player)[hand].mask &
                        board_mask) == 0;
          filtered_range[hand] =
              valid ? opponent_range[hand] *
                          strategy[decision.entry(action, hand)]
                    : 0.0F;
        }
        child_range = filtered_range;
      }

      evaluate_node(children[action].child, player, child_range, child_values,
                    board_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        float action_weight =
            node.player == player
                ? strategy[decision.entry(action, hand)]
                : 1.0F;
        node_values[hand] += action_weight * child_values[hand];
      }
    }
    zero_blocked_values(terminals.hands(player), board_mask, node_values);
    return;
  }
  }
}

void RecursiveEvaluator::cfr_update_node(
    NodeIndex node_index, Player player, std::span<const float> player_reach,
    std::span<const float> opponent_reach, std::vector<float> &node_values,
    float chance_reach, float iteration_weight, uint64_t board_mask) {
  const GameNode &node = tree.nodes[node_index];
  node_values.resize(tree.hand_count_for(player));

  switch (node.type) {
  case NodeType::Fold:
    terminal_fold(node, player, opponent_reach, node_values, board_mask);
    return;

  case NodeType::Showdown:
    terminal_showdown(node, player, opponent_reach, node_values, board_mask);
    return;

  case NodeType::Chance: {
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> child_reach(player_reach.size());
    std::vector<float> child_values;
    for (const GameEdge &edge : tree.children(node)) {
      for (size_t hand = 0; hand < player_reach.size(); ++hand) {
        child_reach[hand] = edge.probability * player_reach[hand];
      }
      cfr_update_node(edge.child, player, child_reach, opponent_reach,
                      child_values, chance_reach * edge.probability,
                      iteration_weight, board_mask | edge.dealt_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        node_values[hand] += edge.probability * child_values[hand];
      }
    }
    return;
  }

  case NodeType::Decision: {
    const DecisionNode &decision = tree.decisions[node.decision_index];
    std::vector<float> strategy(decision.entry_count());
    decision_strategy(decision, false, strategy);
    auto children = tree.children(node);
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> child_values;

    if (node.player == player) {
      std::vector<float> valid_player_reach;
      mask_range(terminals.hands(player), board_mask, player_reach,
                 valid_player_reach);
      tree.state.accumulate_strategy(decision, valid_player_reach,
                                     iteration_weight);
      std::vector<float> action_values(decision.entry_count());
      std::vector<float> child_reach(player_reach.size());

      for (size_t action = 0; action < children.size(); ++action) {
        for (size_t hand = 0; hand < player_reach.size(); ++hand) {
          child_reach[hand] =
              valid_player_reach[hand] *
              strategy[decision.entry(action, hand)];
        }
        cfr_update_node(children[action].child, player, child_reach,
                        opponent_reach, child_values, chance_reach,
                        iteration_weight, board_mask);
        for (size_t hand = 0; hand < node_values.size(); ++hand) {
          size_t entry = decision.entry(action, hand);
          action_values[entry] = child_values[hand];
          node_values[hand] += strategy[entry] * child_values[hand];
        }
      }

      std::vector<float> regret_deltas(decision.entry_count());
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        for (size_t action = 0; action < decision.action_count; ++action) {
          size_t entry = decision.entry(action, hand);
          bool valid =
              (terminals.hands(player)[hand].mask & board_mask) == 0;
          regret_deltas[entry] =
              valid ? chance_reach *
                          (action_values[entry] - node_values[hand])
                    : 0.0F;
        }
      }
      tree.state.apply_regret_deltas(decision, regret_deltas);
      zero_blocked_values(terminals.hands(player), board_mask, node_values);
      return;
    }

    std::vector<float> filtered_range(opponent_reach.size());
    for (size_t action = 0; action < children.size(); ++action) {
      for (size_t hand = 0; hand < opponent_reach.size(); ++hand) {
        bool valid =
            (terminals.hands(node.player)[hand].mask & board_mask) == 0;
        filtered_range[hand] =
            valid ? opponent_reach[hand] *
                        strategy[decision.entry(action, hand)]
                  : 0.0F;
      }
      cfr_update_node(children[action].child, player, player_reach,
                      filtered_range, child_values, chance_reach,
                      iteration_weight, board_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        node_values[hand] += child_values[hand];
      }
    }
    zero_blocked_values(terminals.hands(player), board_mask, node_values);
    return;
  }
  }
}

void RecursiveEvaluator::best_response(Player player,
                                       const Range &opponent_range,
                                       std::vector<float> &values,
                                       bool use_average_strategy) {
  assert(opponent_range.hands ==
         terminals.hand_tables[static_cast<size_t>(opponent(player))]);
  best_response_node(tree.root, player, opponent_range.weights, values,
                     use_average_strategy, 0);
}

float RecursiveEvaluator::best_response_value(Player player,
                                              const Range &player_range,
                                              const Range &opponent_range,
                                              bool use_average_strategy) {
  assert(player_range.hands ==
         terminals.hand_tables[static_cast<size_t>(player)]);
  std::vector<float> values;
  best_response(player, opponent_range, values, use_average_strategy);

  std::vector<float> compatible_weights(player_range.weights.size());
  terminals.apply_fold(player, opponent_range.weights, 1.0F,
                       compatible_weights);
  float total_weight = 0.0F;
  float total_value = 0.0F;
  for (size_t hand = 0; hand < values.size(); ++hand) {
    total_weight += player_range.weights[hand] * compatible_weights[hand];
    total_value += player_range.weights[hand] * values[hand];
  }
  return total_weight > 0.0F ? total_value / total_weight : 0.0F;
}

float RecursiveEvaluator::exploitability(const Range &hero_range,
                                         const Range &villain_range,
                                         bool use_average_strategy) {
  float hero_best =
      best_response_value(Player::Hero, hero_range, villain_range,
                          use_average_strategy);
  float villain_best =
      best_response_value(Player::Villain, villain_range, hero_range,
                          use_average_strategy);
  return 0.5F * (hero_best + villain_best);
}

void RecursiveEvaluator::best_response_node(
    NodeIndex node_index, Player player, std::span<const float> opponent_reach,
    std::vector<float> &node_values, bool use_average_strategy,
    uint64_t board_mask) {
  const GameNode &node = tree.nodes[node_index];
  node_values.resize(tree.hand_count_for(player));

  switch (node.type) {
  case NodeType::Fold:
    terminal_fold(node, player, opponent_reach, node_values, board_mask);
    return;
  case NodeType::Showdown:
    terminal_showdown(node, player, opponent_reach, node_values, board_mask);
    return;
  case NodeType::Chance: {
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> child_values;
    for (const GameEdge &edge : tree.children(node)) {
      best_response_node(edge.child, player, opponent_reach, child_values,
                         use_average_strategy, board_mask | edge.dealt_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        node_values[hand] += edge.probability * child_values[hand];
      }
    }
    return;
  }
  case NodeType::Decision: {
    const DecisionNode &decision = tree.decisions[node.decision_index];
    auto children = tree.children(node);
    std::vector<float> child_values;

    if (node.player == player) {
      node_values.assign(tree.hand_count_for(player),
                         -std::numeric_limits<float>::infinity());
      for (const GameEdge &child : children) {
        best_response_node(child.child, player, opponent_reach, child_values,
                           use_average_strategy, board_mask);
        for (size_t hand = 0; hand < node_values.size(); ++hand) {
          node_values[hand] = std::max(node_values[hand], child_values[hand]);
        }
      }
      zero_blocked_values(terminals.hands(player), board_mask, node_values);
      return;
    }

    std::vector<float> strategy;
    decision_strategy(decision, use_average_strategy, strategy);
    node_values.assign(tree.hand_count_for(player), 0.0F);
    std::vector<float> filtered_range(opponent_reach.size());
    for (size_t action = 0; action < children.size(); ++action) {
      for (size_t hand = 0; hand < opponent_reach.size(); ++hand) {
        bool valid =
            (terminals.hands(node.player)[hand].mask & board_mask) == 0;
        filtered_range[hand] =
            valid ? opponent_reach[hand] *
                        strategy[decision.entry(action, hand)]
                  : 0.0F;
      }
      best_response_node(children[action].child, player, filtered_range,
                         child_values, use_average_strategy, board_mask);
      for (size_t hand = 0; hand < node_values.size(); ++hand) {
        node_values[hand] += child_values[hand];
      }
    }
    zero_blocked_values(terminals.hands(player), board_mask, node_values);
    return;
  }
  }
}

void RecursiveEvaluator::terminal_fold(const GameNode &node, Player player,
                                       std::span<const float> opponent_reach,
                                       std::vector<float> &values,
                                       uint64_t board_mask) const {
  std::span<const float> valid_reach = opponent_reach;
  std::vector<float> masked_reach;
  if (board_mask != 0) {
    mask_range(terminals.hands(opponent(player)), board_mask, opponent_reach,
               masked_reach);
    valid_reach = masked_reach;
  }
  terminals.apply_fold(player, valid_reach,
                       player == Player::Hero ? node.payoff : -node.payoff,
                       values);
  zero_blocked_values(terminals.hands(player), board_mask, values);
}

void RecursiveEvaluator::terminal_showdown(
    const GameNode &node, Player player, std::span<const float> opponent_reach,
    std::vector<float> &values, uint64_t board_mask) const {
  std::span<const float> valid_reach = opponent_reach;
  std::vector<float> masked_reach;
  if (board_mask != 0) {
    mask_range(terminals.hands(opponent(player)), board_mask, opponent_reach,
               masked_reach);
    valid_reach = masked_reach;
  }
  if (player == Player::Hero) {
    terminals.apply_showdown(player, node.runout_index, valid_reach,
                             node.win_payoff, node.loss_payoff, values);
  } else {
    terminals.apply_showdown(player, node.runout_index, valid_reach,
                             -node.loss_payoff, -node.win_payoff, values);
  }
  zero_blocked_values(terminals.hands(player), board_mask, values);
}

void RecursiveEvaluator::decision_strategy(const DecisionNode &decision,
                                           bool use_average_strategy,
                                           std::vector<float> &strategy) const {
  if (use_average_strategy) {
    tree.state.average_strategy(decision, strategy);
  } else {
    strategy.resize(decision.entry_count());
    tree.state.strategy(decision, strategy);
  }
}

} // namespace spmv_poker
