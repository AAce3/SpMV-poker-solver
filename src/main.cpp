#include "spmv_poker/evaluator.h"
#include "spmv_poker/game_config.h"
#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using namespace spmv_poker;

uint8_t card(std::string_view name) {
  constexpr std::string_view ranks = "23456789TJQKA";
  constexpr std::string_view suits = "cdhs";
  return static_cast<uint8_t>(ranks.find(name[0]) * 4 + suits.find(name[1]));
}

Hand hand(std::string_view first, std::string_view second) {
  uint8_t first_card = card(first);
  uint8_t second_card = card(second);
  return {first_card, second_card,
          card_mask(first_card) | card_mask(second_card)};
}

Range range(std::initializer_list<Hand> hands) {
  return {std::vector<Hand>(hands), std::vector<float>(hands.size(), 1.0F)};
}

std::string card_name(uint8_t card) {
  constexpr std::string_view ranks = "23456789TJQKA";
  constexpr std::string_view suits = "cdhs";
  return {ranks[card / 4], suits[card % 4]};
}

std::string hand_name(const Hand &hand) {
  return card_name(hand.first) + card_name(hand.second);
}

void print_strategy(const GameTree &tree, const TerminalTables &terminals,
                    NodeIndex node_index, std::string_view title,
                    std::span<const std::string_view> actions) {
  const GameNode &node = tree.nodes[node_index];
  const DecisionNode &decision = tree.decisions[node.decision_index];
  std::vector<float> average;
  tree.state.average_strategy(decision, average);

  std::cout << "\n" << title << "\n";
  std::cout << std::left << std::setw(8) << "Hand";
  for (std::string_view action : actions) {
    std::cout << std::right << std::setw(11) << action;
  }
  std::cout << '\n';

  for (size_t hand_index = 0; hand_index < decision.hand_count; ++hand_index) {
    std::cout << std::left << std::setw(8)
              << hand_name(terminals.hands(node.player)[hand_index]);
    for (size_t action = 0; action < decision.action_count; ++action) {
      std::cout << std::right << std::setw(10) << std::fixed
                << std::setprecision(1)
                << 100.0F * average[decision.entry(action, hand_index)] << "%";
    }
    std::cout << '\n';
  }
}

} // namespace

int main() {
  using namespace spmv_poker;

  std::array<uint8_t, 5> board{card("2c"), card("7d"), card("9h"), card("Js"),
                               card("Kc")};
  Range hero = range({
      hand("Ts", "8s"), // Straight
      hand("As", "Ah"), // Overpair
      hand("Ks", "Qh"), // Top pair
      hand("5s", "4s"), // Air
  });
  Range villain = range({
      hand("Tc", "8c"), // Straight
      hand("Ad", "Ac"), // Overpair
      hand("Kh", "Qd"), // Top pair
      hand("6c", "3h"), // Air
  });

  TerminalTables terminals(board, hero, villain);
  GameConfig config{.starting_pot = 2.0F, .bet_sizes = {0.5F}};
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  for (size_t iteration = 1; iteration <= 100000; ++iteration) {
    evaluator.cfr_iteration(hero, villain, static_cast<float>(iteration));
  }

  const GameNode &root = tree.nodes[tree.root];
  auto root_children = tree.children(root);
  NodeIndex villain_after_check = root_children[0].child;
  NodeIndex villain_facing_bet = root_children[1].child;
  NodeIndex hero_facing_bet =
      tree.children(tree.nodes[villain_after_check])[1].child;

  std::cout << "River: 2c 7d 9h Js Kc\n";
  std::cout << "Pot: 2.0, bet size: 1.0 (half pot)\n";
  std::cout << "Hero and Villain each hold one straight, overpair, top pair, "
               "and air combo.\n";

  constexpr std::array<std::string_view, 2> root_actions{"Check", "Bet"};
  constexpr std::array<std::string_view, 2> response_actions{"Fold", "Call"};
  print_strategy(tree, terminals, tree.root, "Hero at root", root_actions);
  print_strategy(tree, terminals, villain_facing_bet, "Villain facing Hero bet",
                 response_actions);
  print_strategy(tree, terminals, villain_after_check,
                 "Villain after Hero checks", root_actions);
  print_strategy(tree, terminals, hero_facing_bet, "Hero facing Villain bet",
                 response_actions);
}
