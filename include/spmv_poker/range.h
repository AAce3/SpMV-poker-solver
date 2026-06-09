#pragma once

#include <stddef.h>
#include <vector>

namespace spmv_poker {

struct Range {
    std::vector<float> weights;

    void set_uniform(size_t hand_count);
};

}  // namespace spmv_poker
