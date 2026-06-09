#include "spmv_poker/decision_node.h"

#include <algorithm>
#include <cassert>

namespace spmv_poker {

void DecisionNode::update_strategy() {
  for (size_t hand = 0; hand < hand_count; ++hand) {
    size_t begin = index(hand, 0);
    float positive_regret_sum = 0.0F;
    for (size_t action = 0; action < action_count; ++action) {
      positive_regret_sum += std::max(0.0F, regrets[begin + action]);
    }

    if (positive_regret_sum > 0.0F) {
      for (size_t action = 0; action < action_count; ++action) {
        strategy[begin + action] =
            std::max(0.0F, regrets[begin + action]) / positive_regret_sum;
      }
    } else {
      std::fill_n(strategy.begin() + begin, action_count,
                  1.0F / static_cast<float>(action_count));
    }
  }
}

void DecisionNode::accumulate_strategy(const std::vector<float> &reach_weights,
                                       float iteration_weight) {
  assert(reach_weights.size() == hand_count);

  for (size_t hand = 0; hand < hand_count; ++hand) {
    size_t begin = index(hand, 0);
    float weight = reach_weights[hand] * iteration_weight;
    for (size_t action = 0; action < action_count; ++action) {
      strategy_sum[begin + action] += weight * strategy[begin + action];
    }
  }
}

void DecisionNode::average_strategy(std::vector<float> &average) const {
  average.resize(strategy_sum.size());
  for (size_t hand = 0; hand < hand_count; ++hand) {
    size_t begin = index(hand, 0);
    float total = 0.0F;
    for (size_t action = 0; action < action_count; ++action) {
      total += strategy_sum[begin + action];
    }

    if (total > 0.0F) {
      for (size_t action = 0; action < action_count; ++action) {
        average[begin + action] = strategy_sum[begin + action] / total;
      }
    } else {
      std::fill_n(average.begin() + begin, action_count,
                  1.0F / static_cast<float>(action_count));
    }
  }
}

} // namespace spmv_poker
