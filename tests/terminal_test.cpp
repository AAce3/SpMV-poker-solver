#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

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

Range uniform_range(std::span<const Hand> hands) {
  Range range;
  range.set_uniform(hands);
  return range;
}

std::vector<uint16_t> ranks(const TerminalTables &tables, Player player,
                            size_t runout_index) {
  const RankedHandTable &table = tables.ranked[static_cast<size_t>(player)];
  IndexRange hand_range = table.runout_hand_ranges[runout_index];
  IndexRange group_range = table.runout_group_ranges[runout_index];
  std::vector<uint16_t> result(tables.hands(player).size());
  uint16_t begin = 0;
  for (size_t group_offset = 0; group_offset < group_range.count;
       ++group_offset) {
    const RankGroup &group = table.groups[group_range.begin + group_offset];
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      result[table.hand_indices[hand_range.begin + offset]] = group.rank;
    }
    begin = group.end;
  }
  return result;
}

void apply_showdown_reference(const TerminalTables &tables,
                              Player evaluated_player, size_t runout_index,
                              const Range &opponent_range,
                              std::vector<float> &values) {
  Player opponent_player = opponent(evaluated_player);
  auto evaluated_hands = tables.hands(evaluated_player);
  auto opponent_hands = tables.hands(opponent_player);
  auto evaluated_ranks = ranks(tables, evaluated_player, runout_index);
  auto opponent_ranks = ranks(tables, opponent_player, runout_index);
  values.assign(evaluated_hands.size(), 0.0F);

  for (size_t first = 0; first < evaluated_hands.size(); ++first) {
    for (size_t second = 0; second < opponent_hands.size(); ++second) {
      if (evaluated_ranks[first] == 0 || opponent_ranks[second] == 0 ||
          overlaps(evaluated_hands[first], opponent_hands[second])) {
        continue;
      }
      int payoff = (evaluated_ranks[first] < opponent_ranks[second]) -
                   (evaluated_ranks[first] > opponent_ranks[second]);
      values[first] += opponent_range.weights[second] * payoff;
    }
  }
}

void check_values_match(const TerminalTables &tables, Player evaluated_player,
                        size_t runout_index, const Range &opponent_range) {
  std::vector<float> optimized;
  std::vector<float> reference;
  tables.apply_showdown(evaluated_player, runout_index, opponent_range, 1.0F,
                        -1.0F, optimized);
  apply_showdown_reference(tables, evaluated_player, runout_index,
                           opponent_range, reference);
  float max_error = 0.0F;
  for (size_t index = 0; index < optimized.size(); ++index) {
    max_error =
        std::max(max_error, std::abs(optimized[index] - reference[index]));
  }
  check(max_error < 0.002F, "showdown mismatch");
}

void test_full_tables() {
  TerminalTables river(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  check(river.hands(Player::Hero).size() == 1081,
        "river must leave 1,081 hero hands");
  check(river.hands(Player::Villain).size() == 1081,
        "river must leave 1,081 villain hands");
  check(river.runouts.size() == 1, "river must have one runout");
  check(river.ranked[0].runout_group_ranges.size() == 1,
        "hero has one river ranking");
  check(river.ranked[1].runout_group_ranges.size() == 1,
        "villain has one river ranking");

  TerminalTables turn(std::array<uint8_t, 4>{0, 5, 10, 15});
  check(turn.hands(Player::Hero).size() == 1128,
        "turn must leave 1,128 hands");
  check(turn.runouts.size() == 48, "turn must enumerate every river");

  TerminalTables flop(std::array<uint8_t, 3>{0, 5, 10});
  check(flop.hands(Player::Hero).size() == 1176,
        "flop must leave 1,176 hands");
  check(flop.runouts.size() == 1176,
        "flop must enumerate every turn-river pair");
}

void test_asymmetric_player_tables() {
  std::array<uint8_t, 4> turn{0, 5, 10, 15};
  TerminalTables full(turn);
  Range hero;
  Range villain;
  for (size_t index = 0; index < full.hands(Player::Hero).size(); ++index) {
    if (index % 2 == 0) {
      hero.hands.push_back(full.hands(Player::Hero)[index]);
      hero.weights.push_back(1.0F);
    }
    if (index % 3 == 0) {
      villain.hands.push_back(full.hands(Player::Villain)[index]);
      villain.weights.push_back(static_cast<float>((index % 7) + 1) / 7.0F);
    }
  }
  TerminalTables tables(turn, hero, villain);

  check(std::equal(tables.hands(Player::Hero).begin(),
                   tables.hands(Player::Hero).end(), hero.hands.begin()),
        "hero table uses hero range directly");
  check(std::equal(tables.hands(Player::Villain).begin(),
                   tables.hands(Player::Villain).end(), villain.hands.begin()),
        "villain table uses villain range directly");
  check_values_match(tables, Player::Hero, 0, villain);
  check_values_match(tables, Player::Hero, 27, villain);
  check_values_match(tables, Player::Villain, 47, hero);
}

void test_fold_terminal() {
  TerminalTables tables(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  Range villain = uniform_range(tables.hands(Player::Villain));
  std::vector<float> values;
  tables.apply_fold(Player::Hero, villain, 2.5F, values);
  for (float value : values) {
    check(value == 2.5F * 990.0F,
          "fold utility includes every compatible opponent hand");
  }
}

} // namespace

int main() {
  try {
    test_full_tables();
    test_asymmetric_player_tables();
    test_fold_terminal();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "All terminal tests passed\n";
  return EXIT_SUCCESS;
}
