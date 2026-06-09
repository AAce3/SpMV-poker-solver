#include "spmv_poker/range.h"

namespace spmv_poker {

void Range::set_uniform(size_t hand_count) {
    weights.assign(hand_count, 1.0F);
}

}  // namespace spmv_poker
