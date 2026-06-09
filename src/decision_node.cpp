#include "spmv_poker/decision_node.h"

#include <algorithm>
#include <cassert>

namespace spmv_poker {

void SolverState::update_strategy(const DecisionNode &node) {
  std::span<float> node_regrets = regret_span(node);
  std::span<float> node_strategy = strategy_span(node);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    std::span<float> hand_regrets =
        node_regrets.subspan(hand * node.action_count, node.action_count);
    std::span<float> hand_strategy =
        node_strategy.subspan(hand * node.action_count, node.action_count);
    float positive_regret_sum = 0.0F;
    for (float regret : hand_regrets) {
      positive_regret_sum += std::max(0.0F, regret);
    }

    if (positive_regret_sum > 0.0F) {
      for (size_t action = 0; action < node.action_count; ++action) {
        hand_strategy[action] =
            std::max(0.0F, hand_regrets[action]) / positive_regret_sum;
      }
    } else {
      std::fill(hand_strategy.begin(), hand_strategy.end(),
                1.0F / static_cast<float>(node.action_count));
    }
  }
}

void SolverState::accumulate_strategy(const DecisionNode &node,
                                      std::span<const float> reach_weights,
                                      float iteration_weight) {
  assert(reach_weights.size() == node.hand_count);

  std::span<float> node_strategy = strategy_span(node);
  std::span<float> node_strategy_sum = strategy_sum_span(node);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    float weight = reach_weights[hand] * iteration_weight;
    size_t begin = hand * node.action_count;
    for (size_t action = begin; action < begin + node.action_count; ++action) {
      node_strategy_sum[action] += weight * node_strategy[action];
    }
  }
}

void SolverState::average_strategy(const DecisionNode &node,
                                   std::vector<float> &average) const {
  average.resize(node.entry_count());
  std::span<const float> node_strategy_sum = strategy_sum_span(node);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    size_t begin = hand * node.action_count;
    float total = 0.0F;
    for (size_t action = 0; action < node.action_count; ++action) {
      total += node_strategy_sum[begin + action];
    }

    if (total > 0.0F) {
      for (size_t action = 0; action < node.action_count; ++action) {
        average[begin + action] = node_strategy_sum[begin + action] / total;
      }
    } else {
      std::fill_n(average.begin() + begin, node.action_count,
                  1.0F / static_cast<float>(node.action_count));
    }
  }
}

} // namespace spmv_poker
