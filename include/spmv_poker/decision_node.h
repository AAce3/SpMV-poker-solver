#pragma once

#include "spmv_poker/cards.h"

#include <algorithm>
#include <cassert>
#include <stddef.h>
#include <span>
#include <vector>

namespace spmv_poker {

/**
 * Describes one decision point's slice of the global solver state.
 *
 * Entries within the slice are stored in hand-major order.
 */
struct DecisionNode {
  IndexRange entries;
  size_t hand_count;
  size_t action_count;

  [[nodiscard]] size_t entry_count() const {
    return entries.count;
  }

  [[nodiscard]] size_t entry(size_t action, size_t hand) const {
    assert(action < action_count);
    assert(hand < hand_count);
    return hand * action_count + action;
  }
};

struct SolverState {
  std::vector<float> regrets;
  std::vector<float> strategy_sum;

  [[nodiscard]] std::span<float> regret_span(const DecisionNode &node) {
    return node.entries.view(regrets);
  }

  [[nodiscard]] std::span<float> strategy_sum_span(const DecisionNode &node) {
    return node.entries.view(strategy_sum);
  }

  [[nodiscard]] std::span<const float>
  strategy_sum_span(const DecisionNode &node) const {
    return node.entries.view(strategy_sum);
  }

  void apply_regret_deltas(const DecisionNode &node,
                           std::span<const float> deltas) {
    assert(deltas.size() == node.entry_count());
    auto node_regrets = regret_span(node);
    for (size_t entry = 0; entry < node_regrets.size(); ++entry) {
      node_regrets[entry] =
          std::max(0.0F, node_regrets[entry] + deltas[entry]);
    }
  }

  void strategy(const DecisionNode &node, std::span<float> output) const;
  void accumulate_strategy(const DecisionNode &node,
                           std::span<const float> reach_weights,
                           float iteration_weight = 1.0F);
  void average_strategy(const DecisionNode &node,
                        std::vector<float> &average) const;
};

} // namespace spmv_poker
