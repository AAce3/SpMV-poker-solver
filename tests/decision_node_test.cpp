#include "spmv_poker/game_tree.h"

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
  check(std::abs(actual - expected) < 0.00001F, message);
}

GameTree make_two_decision_tree() {
  GameTree tree(2);
  NodeIndex fold = tree.add_fold_node(-1.0F);
  NodeIndex showdown = tree.add_showdown_node(0, 1.0F, -1.0F);
  std::array children{fold, showdown};
  tree.add_decision_node(Player::Villain, children);
  tree.root = tree.add_decision_node(Player::Hero, children);
  return tree;
}

void test_global_layout() {
  GameTree tree = make_two_decision_tree();
  const DecisionNode &first = tree.decisions[0];
  const DecisionNode &second = tree.decisions[1];

  check(first.entry_count() == 4, "first decision has four entries");
  check(first.entries.begin == 0, "first decision starts at zero");
  check(second.entries.begin == first.entry_count(),
        "decision slices are adjacent");
  check(tree.state.regrets == std::vector<float>(8, 0.0F),
        "global regrets start at zero");
  check(tree.state.strategy_sum == std::vector<float>(8, 0.0F),
        "global strategy sum starts at zero");
  for (float probability : tree.state.strategy) {
    check_close(probability, 0.5F, "decision strategies start uniform");
  }
}

void test_cfr_plus_regret_matching() {
  GameTree tree = make_two_decision_tree();
  DecisionNode &untouched = tree.decisions[0];
  DecisionNode &node = tree.decisions[1];

  std::array deltas{-2.0F, 4.0F, 3.0F, 1.0F};
  tree.state.apply_regret_deltas(node, deltas);
  std::span<float> regrets = tree.state.regret_span(node);
  check(regrets[0] == 0.0F, "CFR+ clamps negative regret");
  check(regrets[1] == 4.0F, "CFR+ accumulates positive regret");

  tree.state.update_strategy(node);
  std::span<float> untouched_strategy = tree.state.strategy_span(untouched);
  std::span<float> strategy = tree.state.strategy_span(node);
  check_close(untouched_strategy[0], 0.5F,
              "other decisions remain untouched");
  check_close(strategy[0], 0.0F, "negative regret gets zero probability");
  check_close(strategy[1], 1.0F, "positive regret gets full probability");
  check_close(strategy[2], 0.75F, "strategy matches positive regrets");
  check_close(strategy[3], 0.25F, "strategy matches positive regrets");
}

void test_average_strategy() {
  GameTree tree = make_two_decision_tree();
  DecisionNode &node = tree.decisions[1];
  std::span<float> regrets = tree.state.regret_span(node);

  std::array first_regrets{1.0F, 3.0F, 3.0F, 1.0F};
  std::copy(first_regrets.begin(), first_regrets.end(), regrets.begin());
  tree.state.update_strategy(node);
  std::array first_reach{1.0F, 0.0F};
  tree.state.accumulate_strategy(node, first_reach);

  std::array second_regrets{3.0F, 1.0F, 1.0F, 3.0F};
  std::copy(second_regrets.begin(), second_regrets.end(), regrets.begin());
  tree.state.update_strategy(node);
  std::array second_reach{1.0F, 4.0F};
  tree.state.accumulate_strategy(node, second_reach, 2.0F);

  std::vector<float> average;
  tree.state.average_strategy(node, average);
  check_close(average[0], 7.0F / 12.0F, "first hand action zero average");
  check_close(average[1], 5.0F / 12.0F, "first hand action one average");
  check_close(average[2], 0.25F, "second hand action zero average");
  check_close(average[3], 0.75F, "second hand action one average");
}

} // namespace

int main() {
  try {
    test_global_layout();
    test_cfr_plus_regret_matching();
    test_average_strategy();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All decision node tests passed\n";
  return EXIT_SUCCESS;
}
