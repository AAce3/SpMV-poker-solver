#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <array>
#include <iostream>

int main() {
    using namespace spmv_poker;

    TerminalTables tables(std::array<uint8_t, 4>{0, 5, 10, 15});
    Range hero_range;
    Range villain_range;
    hero_range.set_uniform(tables.hands(Player::Hero));
    villain_range.set_uniform(tables.hands(Player::Villain));

    std::cout << "Hero range entries: " << hero_range.weights.size() << '\n';
    std::cout << "Villain range entries: " << villain_range.weights.size() << '\n';
    std::cout << "Runouts: " << tables.runouts.size() << '\n';
    std::cout << "Hero rank tables: "
              << tables.ranked[0].runout_group_ranges.size() << '\n';
}
