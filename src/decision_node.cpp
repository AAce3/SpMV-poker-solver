#include "spmv_poker/decision_node.h"

#include <algorithm>
#include <cassert>

namespace spmv_poker {

void SolverState::strategy(const DecisionNode &node,
                           std::span<float> output) const {
  assert(output.size() == node.entry_count());
  auto node_regrets = node.entries.view(regrets);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    size_t begin = node.entry(0, hand);
    auto hand_regrets = node_regrets.subspan(begin, node.action_count);
    auto hand_strategy = output.subspan(begin, node.action_count);
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

  std::vector<float> node_strategy(node.entry_count());
  strategy(node, node_strategy);
  auto node_strategy_sum = strategy_sum_span(node);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    float weight = reach_weights[hand] * iteration_weight;
    size_t begin = node.entry(0, hand);
    for (size_t entry = begin; entry < begin + node.action_count; ++entry) {
      node_strategy_sum[entry] += weight * node_strategy[entry];
    }
  }
}

void SolverState::average_strategy(const DecisionNode &node,
                                   std::vector<float> &average) const {
  average.resize(node.entry_count());
  auto node_strategy_sum = strategy_sum_span(node);
  for (size_t hand = 0; hand < node.hand_count; ++hand) {
    size_t begin = node.entry(0, hand);
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
