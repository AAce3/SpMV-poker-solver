#include "spmv_poker/evaluator.h"

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

void check_close(float actual, float expected, std::string_view message) {
  check(std::abs(actual - expected) < 0.0001F, message);
}

void check_values_close(const std::vector<float> &actual,
                        const std::vector<float> &expected,
                        std::string_view message) {
  check(actual.size() == expected.size(), message);
  for (size_t index = 0; index < actual.size(); ++index) {
    check_close(actual[index], expected[index], message);
  }
}

Range uniform_range(size_t hand_count) {
  Range range;
  range.set_uniform(hand_count);
  return range;
}

void test_fold_perspective() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hand_table.size());
  tree.root = tree.add_fold_node(2.0F);
  Range opponent_range = uniform_range(tree.hand_count);
  Evaluator evaluator{tree, terminals};

  std::vector<float> hero_values;
  std::vector<float> villain_values;
  evaluator.evaluate(Player::Hero, opponent_range, hero_values);
  evaluator.evaluate(Player::Villain, opponent_range, villain_values);

  for (size_t hand = 0; hand < tree.hand_count; ++hand) {
    check_close(hero_values[hand], -villain_values[hand],
                "fold values are zero sum");
  }
}

void test_showdown_dispatch() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hand_table.size());
  tree.root = tree.add_showdown_node(0, 3.0F, -1.0F);
  Range opponent_range = uniform_range(tree.hand_count);
  Evaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_showdown(0, opponent_range, 3.0F, -1.0F, expected);
  check_values_close(actual, expected,
                     "showdown dispatches to terminal tables");
}

void test_chance_weighting() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hand_table.size());
  NodeIndex first = tree.add_fold_node(2.0F);
  NodeIndex second = tree.add_fold_node(-2.0F);
  std::array children{
      GameEdge{.child = first, .probability = 0.75F},
      GameEdge{.child = second, .probability = 0.25F},
  };
  tree.root = tree.add_chance_node(children);
  Range opponent_range = uniform_range(tree.hand_count);
  Evaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(opponent_range, 1.0F, expected);
  check_values_close(actual, expected, "chance node weights child values");
}

void test_player_decision_weighting() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hand_table.size());
  NodeIndex first = tree.add_fold_node(2.0F);
  NodeIndex second = tree.add_fold_node(-2.0F);
  std::array children{first, second};
  tree.root = tree.add_decision_node(Player::Hero, children);
  DecisionNode &decision = tree.decisions[0];
  auto strategy = tree.state.strategy_span(decision);
  for (size_t hand = 0; hand < tree.hand_count; ++hand) {
    strategy[hand * 2] = 0.75F;
    strategy[hand * 2 + 1] = 0.25F;
  }
  Range opponent_range = uniform_range(tree.hand_count);
  Evaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(opponent_range, 1.0F, expected);
  check_values_close(actual, expected, "player decision weights child values");
}

void test_opponent_decision_updates_range() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hand_table.size());
  NodeIndex fold = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{fold, zero};
  tree.root = tree.add_decision_node(Player::Villain, children);
  DecisionNode &decision = tree.decisions[0];
  auto strategy = tree.state.strategy_span(decision);
  std::fill(strategy.begin(), strategy.end(), 0.0F);
  for (size_t hand = 0; hand < tree.hand_count / 2; ++hand) {
    strategy[hand * 2] = 1.0F;
    strategy[hand * 2 + 1] = 0.0F;
  }
  for (size_t hand = tree.hand_count / 2; hand < tree.hand_count; ++hand) {
    strategy[hand * 2] = 0.0F;
    strategy[hand * 2 + 1] = 1.0F;
  }

  Range opponent_range = uniform_range(tree.hand_count);
  Range filtered_range;
  filtered_range.weights.resize(tree.hand_count);
  for (size_t hand = 0; hand < tree.hand_count; ++hand) {
    filtered_range.weights[hand] = strategy[hand * 2];
  }
  Evaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(filtered_range, 2.0F, expected);
  check_values_close(actual, expected,
                     "opponent decision filters opponent range");
}

} // namespace

int main() {
  try {
    test_fold_perspective();
    test_showdown_dispatch();
    test_chance_weighting();
    test_player_decision_weighting();
    test_opponent_decision_updates_range();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All evaluator tests passed\n";
  return EXIT_SUCCESS;
}
