#include "spmv_poker/range.h"
#include "spmv_poker/showdown.h"

#include <array>
#include <iostream>

int main() {
    using namespace spmv_poker;

    ShowdownTables tables(std::array<uint8_t, 4>{0, 5, 10, 15});
    Range hero_range;
    Range villain_range;
    hero_range.set_uniform(tables.hand_table.size());
    villain_range.set_uniform(tables.hand_table.size());

    std::cout << "Canonical turn hands: " << tables.hand_table.size() << '\n';
    std::cout << "Hero range entries: " << hero_range.weights.size() << '\n';
    std::cout << "Villain range entries: " << villain_range.weights.size() << '\n';
    std::cout << "Runouts: " << tables.runouts.size() << '\n';
    std::cout << "Rank tables: " << tables.ranked_hand_indices.size() << '\n';
}
