#include "spmv_poker/range.h"
#include "spmv_poker/showdown.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace spmv_poker;

void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

size_t require_hand(const std::vector<Hand>& hands, uint8_t first, uint8_t second) {
    for (size_t index = 0; index < hands.size(); ++index) {
        if (hands[index].first == first && hands[index].second == second) {
            return index;
        }
    }
    throw std::runtime_error("expected hand in canonical list");
}

void apply_showdown_reference(
    const ShowdownTables& tables,
    size_t runout_index,
    const Range& opponent_range,
    std::vector<float>& values
) {
    size_t hand_count = tables.hand_table.size();
    const auto& hand_indices = tables.ranked_hand_indices[runout_index];
    size_t boundary_begin = tables.runout_group_offsets[runout_index];
    size_t boundary_end =
        runout_index + 1 < tables.runout_group_offsets.size()
            ? tables.runout_group_offsets[runout_index + 1]
            : tables.group_boundaries.size();
    std::vector<uint16_t> ranks(hand_count);
    values.assign(hand_count, 0.0F);

    uint16_t rank = 1;
    for (size_t boundary = boundary_begin; boundary + 1 < boundary_end; ++boundary, ++rank) {
        for (
            size_t index = tables.group_boundaries[boundary];
            index < tables.group_boundaries[boundary + 1];
            ++index
        ) {
            ranks[hand_indices[index]] = rank;
        }
    }

    for (size_t first = 0; first < hand_count; ++first) {
        for (size_t second = 0; second < hand_count; ++second) {
            bool compatible =
                (ranks[first] != 0)
                & (ranks[second] != 0)
                & !overlaps(tables.hand_table[first], tables.hand_table[second]);
            int payoff =
                (ranks[first] < ranks[second])
                - (ranks[first] > ranks[second]);
            values[first] += opponent_range.weights[second] * compatible * payoff;
        }
    }
}

void check_values_match(
    const ShowdownTables& tables,
    size_t runout_index,
    const Range& range
) {
    std::vector<float> optimized;
    std::vector<float> reference;
    tables.apply_showdown(runout_index, range, optimized);
    apply_showdown_reference(tables, runout_index, range, reference);

    float max_error = 0.0F;
    for (size_t index = 0; index < optimized.size(); ++index) {
        max_error = std::max(max_error, std::abs(optimized[index] - reference[index]));
    }
    check(max_error < 0.002F, "showdown mismatch");
}

void test_river_runout() {
    ShowdownTables tables(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
    check(tables.hand_table.size() == 1081, "river must leave 1,081 hands");
    check(tables.runouts.size() == 1, "river must have one runout");
    check(tables.ranked_hand_indices.size() == 1, "river must have one rank table");
    check(tables.ranked_hand_indices[0].size() == 1081, "all river hands must be ranked");
    check(tables.runout_group_offsets == std::vector<uint32_t>{0}, "river groups start at zero");
    check(tables.group_boundaries.front() == 0, "groups must start at the first ranked hand");
    check(tables.group_boundaries.back() == 1081, "groups must end after all ranked hands");
}

void test_turn_runouts() {
    ShowdownTables tables(std::array<uint8_t, 4>{0, 5, 10, 15});
    check(tables.hand_table.size() == 1128, "turn must leave 1,128 canonical hands");
    check(tables.runouts.size() == 48, "turn must enumerate every legal river");
    check(tables.ranked_hand_indices.size() == 48, "each river must have a rank table");
    check(tables.runouts.front() == std::array<uint8_t, 5>{0, 5, 10, 15, 1}, "first river");
    check(tables.runouts.back() == std::array<uint8_t, 5>{0, 5, 10, 15, 51}, "last river");
}

void test_flop_runouts() {
    ShowdownTables tables(std::array<uint8_t, 3>{0, 5, 10});
    check(tables.hand_table.size() == 1176, "flop must leave 1,176 canonical hands");
    check(tables.runouts.size() == 1176, "flop must enumerate unordered turn-river pairs");
    check(tables.ranked_hand_indices.size() == 1176, "each flop runout must have a rank table");
    check(tables.runouts.front() == std::array<uint8_t, 5>{0, 5, 10, 1, 2}, "first pair");
    check(tables.runouts.back() == std::array<uint8_t, 5>{0, 5, 10, 50, 51}, "last pair");
}

void test_linear_showdown() {
    ShowdownTables tables(std::array<uint8_t, 4>{0, 5, 10, 15});
    Range range;
    range.weights.resize(tables.hand_table.size());
    for (size_t index = 0; index < range.weights.size(); ++index) {
        range.weights[index] = static_cast<float>((index * 17) % 23) / 23.0F;
    }

    check_values_match(tables, 0, range);
    check_values_match(tables, 27, range);
    check_values_match(tables, 47, range);

    size_t aces = require_hand(tables.hand_table, 49, 51);
    size_t kings = require_hand(tables.hand_table, 45, 47);
    range.weights.assign(tables.hand_table.size(), 0.0F);
    range.weights[kings] = 0.75F;

    std::vector<float> values;
    tables.apply_showdown(27, range, values);
    check(values[aces] == 0.75F, "aces must gain the weight of kings");
}

}  // namespace

int main() {
    try {
        test_river_runout();
        test_turn_runouts();
        test_flop_runouts();
        test_linear_showdown();
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All showdown tests passed\n";
    return EXIT_SUCCESS;
}
