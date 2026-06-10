#include "spmv_poker/evaluator.h"
#include "spmv_poker/game_config.h"
#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <array>
#include <iostream>

namespace {

spmv_poker::Range sample_range(std::span<const spmv_poker::Hand> hands,
                               size_t offset, size_t stride, size_t count) {
  spmv_poker::Range range;
  for (size_t index = offset; index < hands.size() && range.hands.size() < count;
       index += stride) {
    range.hands.push_back(hands[index]);
    range.weights.push_back(1.0F);
  }
  return range;
}

} // namespace

int main() {
  using namespace spmv_poker;

  std::array<uint8_t, 4> board{0, 5, 10, 15};
  TerminalTables full(board);
  Range hero_range = sample_range(full.hands(Player::Hero), 0, 31, 8);
  Range villain_range = sample_range(full.hands(Player::Villain), 13, 37, 8);
  TerminalTables terminals(board, hero_range, villain_range);
  GameConfig config{
      .starting_street = StartingStreet::Turn,
      .starting_pot = 2.0F,
      .bet_sizes = {0.5F, 1.0F},
  };
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  float initial = evaluator.exploitability(hero_range, villain_range);
  for (size_t iteration = 1; iteration <= 1000; ++iteration) {
    evaluator.cfr_iteration(hero_range, villain_range,
                            static_cast<float>(iteration));
  }
  float final = evaluator.exploitability(hero_range, villain_range);

  std::cout << "Simplified turn board: {0, 5, 10, 15}\n";
  std::cout << "Ranges: " << hero_range.weights.size() << " hero hands, "
            << villain_range.weights.size() << " villain hands\n";
  std::cout << "Bet sizes: 0.5 pot, 1.0 pot\n";
  std::cout << "Public nodes: " << tree.nodes.size() << '\n';
  std::cout << "Initial exploitability: " << initial << '\n';
  std::cout << "Exploitability after 1000 iterations: " << final << '\n';
}
