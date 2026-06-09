#pragma once

#include <algorithm>
#include <cassert>
#include <stddef.h>
#include <vector>

namespace spmv_poker {

/**
 * Strategy and regret state for one decision point
 *
 * Entries are stored in compact, hand-major order, i.e.
 * index = hand_index * action_count + action_index
 */
struct DecisionNode {
  size_t hand_count;
  size_t action_count;
  std::vector<float> regrets;
  std::vector<float> strategy;
  std::vector<float> strategy_sum;

  DecisionNode(size_t hand_count, size_t action_count)
      : hand_count(hand_count), action_count(action_count),
        regrets(hand_count * action_count, 0.0F),
        strategy(hand_count * action_count),
        strategy_sum(hand_count * action_count, 0.0F) {
    assert(action_count > 0);
    std::fill(strategy.begin(), strategy.end(),
              1.0F / static_cast<float>(action_count));
  }

  [[nodiscard]] size_t index(size_t hand_index, size_t action_index) const {
    return hand_index * action_count + action_index;
  }

  // use CFR+ to compute regrets
  void apply_regret_deltas(const std::vector<float> &deltas) {
    assert(deltas.size() == regrets.size());
    for (size_t entry = 0; entry < regrets.size(); ++entry) {
      regrets[entry] = std::max(0.0F, regrets[entry] + deltas[entry]);
    }
  }
  void update_strategy();
  void accumulate_strategy(const std::vector<float> &reach_weights,
                           float iteration_weight = 1.0F);
  void average_strategy(std::vector<float> &average) const;
};

} // namespace spmv_poker
