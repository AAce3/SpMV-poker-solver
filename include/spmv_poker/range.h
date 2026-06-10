#pragma once

#include "spmv_poker/cards.h"

#include <stddef.h>
#include <span>
#include <vector>

namespace spmv_poker {

struct Range {
    std::vector<Hand> hands;
    std::vector<float> weights;

    void set_uniform(std::span<const Hand> hand_table);
};

}  // namespace spmv_poker
