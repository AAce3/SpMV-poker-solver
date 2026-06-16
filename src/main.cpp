#include "spmv_poker/spmv_evaluator.h"
#include "spmv_poker/spmv_terminal.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace spmv_poker;

uint8_t card(std::string_view name) {
  constexpr std::string_view ranks = "23456789TJQKA";
  constexpr std::string_view suits = "cdhs";
  return static_cast<uint8_t>(ranks.find(name[0]) * 4 + suits.find(name[1]));
}

Hand make_hand(std::string_view first, std::string_view second) {
  uint8_t first_card = card(first);
  uint8_t second_card = card(second);
  return {first_card, second_card,
          card_mask(first_card) | card_mask(second_card)};
}

Range make_range(std::initializer_list<Hand> hands) {
  return {std::vector<Hand>(hands), std::vector<float>(hands.size(), 1.0F)};
}

std::string card_name(uint8_t value) {
  constexpr std::string_view ranks = "23456789TJQKA";
  constexpr std::string_view suits = "cdhs";
  return {ranks[value / 4], suits[value % 4]};
}

std::string hand_name(const Hand &value) {
  return card_name(value.first) + card_name(value.second);
}

StreetTopology make_demo_river_topology(size_t hero_hand_count,
                                        size_t villain_hand_count) {
  StreetTopology topology(Street::River, hero_hand_count, villain_hand_count);
  NodeIndex showdown = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex hero_bet_call = topology.add_showdown_node(1.5F, -1.5F);
  std::array villain_children{showdown, hero_bet_call};
  NodeIndex villain_after_check =
      topology.add_player_node(Player::Villain, villain_children);
  std::array hero_children{villain_after_check, hero_bet_call};
  topology.root = topology.add_player_node(Player::Hero, hero_children);
  return topology;
}

void print_average_strategy(const StreetTree &tree, BoardIndex board,
                            uint32_t node_slot, std::string_view title,
                            std::span<const std::string_view> actions,
                            std::span<const Hand> hands) {
  const CompiledBackwardNode &node = tree.compiled.nodes[node_slot];
  std::span<const float> cumulative = tree.board_cumulative_strategy(board);
  size_t hand_count = hands.size();
  size_t hand_stride = tree.compiled.padded_hand_counts[static_cast<size_t>(
      node.player)];

  std::cout << "\n" << title << "\n";
  std::cout << std::left << std::setw(8) << "Hand";
  for (std::string_view action : actions) {
    std::cout << std::right << std::setw(11) << action;
  }
  std::cout << '\n';

  for (size_t hand = 0; hand < hand_count; ++hand) {
    std::cout << std::left << std::setw(8) << hand_name(hands[hand]);
    size_t base = tree.state_entry(board, node_slot, hand, 0);
    float total = 0.0F;
    for (size_t action = 0; action < node.action_count; ++action) {
      total += cumulative[base + action * hand_stride];
    }
    for (size_t action = 0; action < node.action_count; ++action) {
      float average = total > 0.0F
                          ? cumulative[base + action * hand_stride] /
                                total
                          : 1.0F / static_cast<float>(node.action_count);
      std::cout << std::right << std::setw(10) << std::fixed
                << std::setprecision(1) << 100.0F * average << "%";
    }
    std::cout << '\n';
  }
}

} // namespace

int main() {
  using namespace spmv_poker;

  std::array<uint8_t, 5> board{card("2c"), card("7d"), card("9h"), card("Js"),
                               card("Kc")};
  Range hero = make_range({
      make_hand("Ts", "8s"), // Straight
      make_hand("As", "Ah"), // Overpair
      make_hand("Ks", "Qh"), // Top pair
      make_hand("5s", "4s"), // Air
  });
  Range villain = make_range({
      make_hand("Tc", "8c"), // Straight
      make_hand("Ad", "Ac"), // Overpair
      make_hand("Kh", "Qd"), // Top pair
      make_hand("6c", "3h"), // Air
  });

  TerminalTables terminals(board, hero, villain);
  StreetTopology topology =
      make_demo_river_topology(hero.hands.size(), villain.hands.size());
  StreetTree tree(std::move(topology), make_mask(board), board.size());
  RankSummaryTerminalOperator terminals_op(terminals, make_mask(board),
                                           board.size());

  std::array<std::vector<float>, 2> roots{
      std::vector<float>(tree.compiled.padded_hand_counts[0], 0.0F),
      std::vector<float>(tree.compiled.padded_hand_counts[1], 0.0F),
  };
  std::copy(hero.weights.begin(), hero.weights.end(), roots[0].begin());
  std::copy(villain.weights.begin(), villain.weights.end(), roots[1].begin());

  SolveProgram program{
      .board_index = tree.board_index,
      .river = tree.compiled,
      .schedule = ExecutionSchedule{},
      .terminals = &terminals_op,
  };
  SolveState state{
      .river = {.regrets = tree.regrets, .cumulative_strategy =
                                        tree.cumulative_strategy},
  };
  CpuSolveWorkspace workspace;
  workspace.river.prepare(tree.compiled, 1);
  CpuSolveExecutor executor;
  std::vector<float> root_values(tree.compiled.padded_hand_counts[0], 0.0F);
  std::array<std::span<const float>, 2> root_ranges{roots[0], roots[1]};

  for (size_t iteration = 1; iteration <= 100000; ++iteration) {
    executor.update_player(program, state, workspace, Player::Hero, root_ranges,
                           root_values, static_cast<float>(iteration));
  }

  StreetTree solved_tree(program.river, *program.board_index, std::move(state.river.regrets),
                        std::move(state.river.cumulative_strategy));

  std::array<std::string_view, 2> root_actions{"Check", "Bet"};
  std::cout << "River: 2c 7d 9h Js Kc\n";
  std::cout << "Sample range: 4 hands per player\n";
  std::cout << "Average strategy at the root after 100000 iterations\n";
  print_average_strategy(solved_tree, 0, solved_tree.compiled.root_value_slot, "Hero root",
                         root_actions, hero.hands);

  std::cout << "\nRoot values\n";
  for (size_t hand = 0; hand < hero.hands.size(); ++hand) {
    std::cout << std::left << std::setw(8) << hand_name(hero.hands[hand])
              << std::right << std::setw(12) << std::fixed
              << std::setprecision(3) << root_values[hand] << '\n';
  }

  return EXIT_SUCCESS;
}
