#include "spmv_poker/range.h"

namespace spmv_poker {

void Range::set_uniform(std::span<const Hand> hand_table) {
    hands.assign(hand_table.begin(), hand_table.end());
    weights.assign(hands.size(), 1.0F);
}

}  // namespace spmv_poker
