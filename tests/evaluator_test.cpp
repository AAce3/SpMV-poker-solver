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

Range uniform_range(std::span<const Hand> hands) {
  Range range;
  range.set_uniform(hands);
  return range;
}

void test_fold_perspective() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  tree.root = tree.add_fold_node(2.0F);
  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> hero_values;
  std::vector<float> villain_values;
  evaluator.evaluate(Player::Hero, opponent_range, hero_values);
  evaluator.evaluate(Player::Villain, opponent_range, villain_values);

  for (size_t hand = 0; hand < tree.hand_count_for(Player::Hero); ++hand) {
    check_close(hero_values[hand], -villain_values[hand],
                "fold values are zero sum");
  }
}

void test_showdown_dispatch() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  tree.root = tree.add_showdown_node(0, 3.0F, -1.0F);
  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_showdown(Player::Hero, 0, opponent_range, 3.0F, -1.0F,
                           expected);
  check_values_close(actual, expected,
                     "showdown dispatches to terminal tables");
}

void test_chance_weighting() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex first = tree.add_fold_node(2.0F);
  NodeIndex second = tree.add_fold_node(-2.0F);
  std::array children{
      GameEdge{.child = first, .probability = 0.75F},
      GameEdge{.child = second, .probability = 0.25F},
  };
  tree.root = tree.add_chance_node(children);
  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(Player::Hero, opponent_range, 1.0F, expected);
  check_values_close(actual, expected, "chance node weights child values");
}

void test_player_decision_weighting() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex first = tree.add_fold_node(2.0F);
  NodeIndex second = tree.add_fold_node(-2.0F);
  std::array children{first, second};
  tree.root = tree.add_decision_node(Player::Hero, children);
  DecisionNode &decision = tree.decisions[0];
  auto regrets = tree.state.regret_span(decision);
  for (size_t hand = 0; hand < tree.hand_count_for(Player::Hero); ++hand) {
    regrets[hand * 2] = 0.75F;
    regrets[hand * 2 + 1] = 0.25F;
  }
  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(Player::Hero, opponent_range, 1.0F, expected);
  check_values_close(actual, expected, "player decision weights child values");
}

void test_opponent_decision_updates_range() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex fold = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{fold, zero};
  tree.root = tree.add_decision_node(Player::Villain, children);
  DecisionNode &decision = tree.decisions[0];
  auto regrets = tree.state.regret_span(decision);
  size_t villain_hand_count = tree.hand_count_for(Player::Villain);
  for (size_t hand = 0; hand < villain_hand_count / 2; ++hand) {
    regrets[hand * 2] = 1.0F;
    regrets[hand * 2 + 1] = 0.0F;
  }
  for (size_t hand = villain_hand_count / 2; hand < villain_hand_count; ++hand) {
    regrets[hand * 2] = 0.0F;
    regrets[hand * 2 + 1] = 1.0F;
  }

  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  Range filtered_range;
  filtered_range.hands = opponent_range.hands;
  filtered_range.weights.resize(villain_hand_count);
  for (size_t hand = 0; hand < villain_hand_count; ++hand) {
    filtered_range.weights[hand] = regrets[hand * 2];
  }
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, opponent_range, actual);
  terminals.apply_fold(Player::Hero, filtered_range, 2.0F, expected);
  check_values_close(actual, expected,
                     "opponent decision filters opponent range");
}

void test_reuses_depth_scratch() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex first = tree.add_fold_node(2.0F);
  NodeIndex second = tree.add_fold_node(-2.0F);
  std::array children{first, second};
  tree.root = tree.add_decision_node(Player::Hero, children);
  Range opponent_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> values;
  evaluator.evaluate(Player::Hero, opponent_range, values);

  check(evaluator.value_scratch.size() ==
            2 * tree.hand_count_for(Player::Hero),
        "evaluator stores values by recursion depth");
  check(evaluator.reach_scratch.size() ==
            2 * tree.hand_count_for(Player::Villain),
        "evaluator stores reach by recursion depth");
}

void test_player_local_ranges() {
  std::array<uint8_t, 5> river{0, 5, 10, 15, 28};
  TerminalTables full(river);
  Range hero;
  Range villain;
  for (size_t hand = 0; hand < full.hands(Player::Hero).size(); ++hand) {
    if (hand % 2 == 0) {
      hero.hands.push_back(full.hands(Player::Hero)[hand]);
      hero.weights.push_back(1.0F);
    }
    if (hand % 3 == 0) {
      villain.hands.push_back(full.hands(Player::Villain)[hand]);
      villain.weights.push_back(1.0F);
    }
  }
  TerminalTables terminals(river, hero, villain);
  GameTree tree(hero.hands.size(), villain.hands.size());
  NodeIndex fold = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{fold, zero};
  tree.root = tree.add_decision_node(Player::Villain, children);
  auto regrets = tree.state.regret_span(tree.decisions[0]);
  for (size_t hand = 0; hand < villain.weights.size(); ++hand) {
    regrets[hand * 2] = hand % 2 == 0 ? 1.0F : 0.0F;
    regrets[hand * 2 + 1] = hand % 2 == 0 ? 0.0F : 1.0F;
  }
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> local_values;
  evaluator.evaluate(Player::Hero, villain, local_values);

  Range filtered_villain = villain;
  for (size_t hand = 0; hand < filtered_villain.weights.size(); ++hand) {
    filtered_villain.weights[hand] = hand % 2 == 0 ? 1.0F : 0.0F;
  }
  std::vector<float> expected;
  terminals.apply_fold(Player::Hero, filtered_villain, 2.0F, expected);

  check(local_values.size() == hero.weights.size(),
        "evaluator returns evaluated player's local hands");
  check_values_close(local_values, expected,
                     "player-local evaluation matches terminal evaluation");
}

} // namespace

int main() {
  try {
    test_fold_perspective();
    test_showdown_dispatch();
    test_chance_weighting();
    test_player_decision_weighting();
    test_opponent_decision_updates_range();
    test_reuses_depth_scratch();
    test_player_local_ranges();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All recursive evaluator tests passed\n";
  return EXIT_SUCCESS;
}
