#include "spmv_poker/evaluator.h"
#include "spmv_poker/game_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
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

Range first_hands(std::span<const Hand> hands, size_t count) {
  Range range;
  range.hands.assign(hands.begin(), hands.begin() + count);
  range.weights.assign(count, 1.0F);
  return range;
}

Range last_hands(std::span<const Hand> hands, size_t count) {
  Range range;
  range.hands.assign(hands.end() - count, hands.end());
  range.weights.assign(count, 1.0F);
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

void test_cfr_update_player_decision() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex win = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{win, zero};
  tree.root = tree.add_decision_node(Player::Hero, children);
  const DecisionNode &decision = tree.decisions[0];
  Range hero_range = uniform_range(terminals.hands(Player::Hero));
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> win_values;
  terminals.apply_fold(Player::Hero, villain_range, 2.0F, win_values);
  std::vector<float> actual;
  evaluator.cfr_update(Player::Hero, hero_range, villain_range, actual);

  auto regrets = tree.state.regret_span(decision);
  auto strategy_sum = tree.state.strategy_sum_span(decision);
  for (size_t hand = 0; hand < hero_range.hands.size(); ++hand) {
    check_close(actual[hand], 0.5F * win_values[hand],
                "CFR returns the pre-update strategy value");
    check_close(regrets[hand * 2], 0.5F * win_values[hand],
                "better action gains regret");
    check_close(regrets[hand * 2 + 1], 0.0F,
                "CFR+ clamps worse action regret");
    check_close(strategy_sum[hand * 2], 0.5F,
                "average strategy accumulates current action zero");
    check_close(strategy_sum[hand * 2 + 1], 0.5F,
                "average strategy accumulates current action one");
  }

  evaluator.cfr_update(Player::Hero, hero_range, villain_range, actual);
  for (size_t hand = 0; hand < hero_range.hands.size(); ++hand) {
    check_close(actual[hand], win_values[hand],
                "next CFR update uses updated strategy");
    check_close(strategy_sum[hand * 2], 1.5F,
                "average strategy accumulates updated action zero");
    check_close(strategy_sum[hand * 2 + 1], 0.5F,
                "unused action retains previous strategy mass");
  }
}

void test_cfr_update_only_updates_selected_player() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex hero_win = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{hero_win, zero};
  tree.root = tree.add_decision_node(Player::Villain, children);
  const DecisionNode &decision = tree.decisions[0];
  Range hero_range = uniform_range(terminals.hands(Player::Hero));
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> values;
  evaluator.cfr_update(Player::Hero, hero_range, villain_range, values);
  check(tree.state.regrets == std::vector<float>(decision.entry_count(), 0.0F),
        "Hero traversal does not update Villain regrets");
  check(tree.state.strategy_sum ==
            std::vector<float>(decision.entry_count(), 0.0F),
        "Hero traversal does not update Villain strategy sum");

  std::vector<float> fold_values;
  terminals.apply_fold(Player::Villain, hero_range, -2.0F, fold_values);
  evaluator.cfr_update(Player::Villain, villain_range, hero_range, values);
  auto regrets = tree.state.regret_span(decision);
  auto strategy_sum = tree.state.strategy_sum_span(decision);
  for (size_t hand = 0; hand < villain_range.hands.size(); ++hand) {
    check_close(values[hand], 0.5F * fold_values[hand],
                "Villain CFR value uses Villain perspective");
    check_close(regrets[hand * 2], 0.0F,
                "losing action regret is clamped");
    check_close(regrets[hand * 2 + 1], -0.5F * fold_values[hand],
                "better Villain action gains regret");
    check_close(strategy_sum[hand * 2], 0.5F,
                "Villain average strategy accumulates action zero");
    check_close(strategy_sum[hand * 2 + 1], 0.5F,
                "Villain average strategy accumulates action one");
  }
}

void test_cfr_chance_weights_strategy_sum() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex first_terminal = tree.add_fold_node(1.0F);
  NodeIndex second_terminal = tree.add_fold_node(0.0F);
  std::array decision_children{first_terminal, second_terminal};
  NodeIndex first_decision =
      tree.add_decision_node(Player::Hero, decision_children);
  NodeIndex second_decision =
      tree.add_decision_node(Player::Hero, decision_children);
  std::array chance_children{
      GameEdge{.child = first_decision, .probability = 0.25F},
      GameEdge{.child = second_decision, .probability = 0.75F},
  };
  tree.root = tree.add_chance_node(chance_children);
  Range hero_range = uniform_range(terminals.hands(Player::Hero));
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> values;
  evaluator.cfr_update(Player::Hero, hero_range, villain_range, values, 2.0F);

  auto first_sum = tree.state.strategy_sum_span(tree.decisions[0]);
  auto second_sum = tree.state.strategy_sum_span(tree.decisions[1]);
  auto first_regrets = tree.state.regret_span(tree.decisions[0]);
  auto second_regrets = tree.state.regret_span(tree.decisions[1]);
  std::vector<float> win_values;
  terminals.apply_fold(Player::Hero, villain_range, 1.0F, win_values);
  for (size_t hand = 0; hand < hero_range.hands.size(); ++hand) {
    check_close(first_sum[hand * 2], 0.25F,
                "first chance child weights average strategy");
    check_close(first_sum[hand * 2 + 1], 0.25F,
                "first chance child weights every action");
    check_close(second_sum[hand * 2], 0.75F,
                "second chance child weights average strategy");
    check_close(second_sum[hand * 2 + 1], 0.75F,
                "second chance child weights every action");
    check_close(first_regrets[hand * 2], 0.125F * win_values[hand],
                "first chance child weights regret");
    check_close(second_regrets[hand * 2], 0.375F * win_values[hand],
                "second chance child weights regret");
  }
}

void test_cfr_iteration_updates_both_players() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex win = tree.add_fold_node(1.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array decision_children{win, zero};
  NodeIndex hero_decision =
      tree.add_decision_node(Player::Hero, decision_children);
  NodeIndex villain_decision =
      tree.add_decision_node(Player::Villain, decision_children);
  std::array chance_children{
      GameEdge{.child = hero_decision, .probability = 0.5F},
      GameEdge{.child = villain_decision, .probability = 0.5F},
  };
  tree.root = tree.add_chance_node(chance_children);
  Range hero_range = uniform_range(terminals.hands(Player::Hero));
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  evaluator.cfr_iteration(hero_range, villain_range);

  auto hero_sum = tree.state.strategy_sum_span(tree.decisions[0]);
  auto villain_sum = tree.state.strategy_sum_span(tree.decisions[1]);
  for (size_t hand = 0; hand < hero_range.hands.size(); ++hand) {
    check_close(hero_sum[hand * 2], 0.25F,
                "CFR iteration accumulates Hero strategy");
    check_close(hero_sum[hand * 2 + 1], 0.25F,
                "CFR iteration accumulates every Hero action");
    check_close(villain_sum[hand * 2], 0.25F,
                "CFR iteration accumulates Villain strategy");
    check_close(villain_sum[hand * 2 + 1], 0.25F,
                "CFR iteration accumulates every Villain action");
  }
}

void test_best_response_selects_best_action() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  NodeIndex win = tree.add_fold_node(2.0F);
  NodeIndex zero = tree.add_fold_node(0.0F);
  std::array children{win, zero};
  tree.root = tree.add_decision_node(Player::Hero, children);
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.best_response(Player::Hero, villain_range, actual);
  terminals.apply_fold(Player::Hero, villain_range, 2.0F, expected);
  check_values_close(actual, expected,
                     "best response selects the best action per hand");
}

void test_terminal_game_has_zero_exploitability() {
  TerminalTables terminals(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());
  tree.root = tree.add_showdown_node(0, 1.0F, -1.0F);
  Range hero_range = uniform_range(terminals.hands(Player::Hero));
  Range villain_range = uniform_range(terminals.hands(Player::Villain));
  RecursiveEvaluator evaluator{tree, terminals};

  check_close(evaluator.exploitability(hero_range, villain_range), 0.0F,
              "game without decisions has zero exploitability");
}

void test_deal_node_defers_blockers_to_terminals() {
  std::array<uint8_t, 4> turn{0, 5, 10, 15};
  TerminalTables full(turn);
  Range hero = first_hands(full.hands(Player::Hero), 8);
  Range villain = first_hands(full.hands(Player::Villain), 8);
  TerminalTables terminals(turn, hero, villain);
  GameTree tree(hero.hands.size(), villain.hands.size());
  std::vector<GameEdge> runouts;
  for (size_t runout = 0; runout < terminals.runouts.size(); ++runout) {
    NodeIndex fold = tree.add_fold_node(1.0F);
    runouts.push_back(GameEdge{.child = fold,
                               .probability = 1.0F / 44.0F,
                               .dealt_mask =
                                   card_mask(terminals.runouts[runout][4])});
  }
  tree.root = tree.add_card_deal_node(runouts);
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> actual;
  std::vector<float> expected;
  evaluator.evaluate(Player::Hero, villain, actual);
  terminals.apply_fold(Player::Hero, villain, 1.0F, expected);
  check_values_close(actual, expected,
                     "deal reduction counts exactly 44 legal rivers per pair");
}

void test_game_config_builds_trees() {
  TerminalTables river(std::array<uint8_t, 5>{0, 5, 10, 15, 28});
  GameConfig single_size;
  GameTree river_tree = build_game_tree(river, single_size);
  check(river_tree.nodes.size() == 9, "river fixture has nine public nodes");
  check(river_tree.decisions.size() == 4,
        "river fixture has four decisions");

  GameConfig multi_size{.bet_sizes = {0.5F, 1.0F}};
  GameTree multi_size_tree = build_game_tree(river, multi_size);
  check(multi_size_tree.nodes.size() == 15,
        "two-size river fixture has fifteen public nodes");
  check(multi_size_tree.decisions.size() == 6,
        "two-size river fixture has six decisions");
  check(multi_size_tree.decisions.back().action_count == 3,
        "two-size river root has check and two bets");

  TerminalTables turn(std::array<uint8_t, 4>{0, 5, 10, 15});
  GameConfig turn_config{.starting_street = StartingStreet::Turn};
  GameTree turn_tree = build_game_tree(turn, turn_config);
  check(turn_tree.nodes[turn_tree.root].type == NodeType::Chance,
        "turn fixture starts with a chance node");
  check(turn_tree.nodes[turn_tree.root].chance_kind == ChanceKind::CardDeal,
        "turn fixture chance node represents a card deal");
  check(turn_tree.children(turn_tree.nodes[turn_tree.root]).size() == 48,
        "turn fixture has every river child");
}

void test_turn_fixture_skips_blocked_decision_hands() {
  std::array<uint8_t, 4> board{0, 5, 10, 15};
  TerminalTables full(board);
  Range hero = first_hands(full.hands(Player::Hero), 8);
  Range villain = last_hands(full.hands(Player::Villain), 8);
  TerminalTables terminals(board, hero, villain);
  GameConfig config{.starting_street = StartingStreet::Turn};
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  std::vector<float> values;
  evaluator.cfr_update(Player::Hero, hero, villain, values);

  bool checked_blocked_hand = false;
  std::function<void(NodeIndex, uint64_t)> check_subtree =
      [&](NodeIndex node_index, uint64_t board_mask) {
        const GameNode &node = tree.nodes[node_index];
        if (node.type == NodeType::Decision && node.player == Player::Hero) {
          const DecisionNode &decision = tree.decisions[node.decision_index];
          auto strategy_sum = tree.state.strategy_sum_span(decision);
          for (size_t hand = 0; hand < hero.hands.size(); ++hand) {
            if ((hero.hands[hand].mask & board_mask) == 0) {
              continue;
            }
            checked_blocked_hand = true;
            for (size_t action = 0; action < decision.action_count; ++action) {
              check_close(
                  strategy_sum[hand * decision.action_count + action], 0.0F,
                  "blocked runout hand accumulates no strategy");
            }
          }
        }
        for (const GameEdge &edge : tree.children(node)) {
          check_subtree(edge.child, board_mask | edge.dealt_mask);
        }
      };
  check_subtree(tree.root, 0);
  check(checked_blocked_hand,
        "turn fixture contains blocked hands at river decisions");
}

void test_river_cfr_reduces_exploitability() {
  std::array<uint8_t, 5> board{0, 5, 10, 15, 28};
  TerminalTables full(board);
  Range hero = first_hands(full.hands(Player::Hero), 6);
  Range villain = last_hands(full.hands(Player::Villain), 6);
  TerminalTables terminals(board, hero, villain);
  GameConfig config;
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  float initial = evaluator.exploitability(hero, villain);
  for (size_t iteration = 1; iteration <= 500; ++iteration) {
    evaluator.cfr_iteration(hero, villain, static_cast<float>(iteration));
  }
  float final = evaluator.exploitability(hero, villain);

  check(final < initial * 0.25F,
        "river CFR substantially reduces average-strategy exploitability");
}

void test_multi_size_river_cfr_reduces_exploitability() {
  std::array<uint8_t, 5> board{0, 5, 10, 15, 28};
  TerminalTables full(board);
  Range hero = first_hands(full.hands(Player::Hero), 8);
  Range villain = last_hands(full.hands(Player::Villain), 8);
  TerminalTables terminals(board, hero, villain);
  GameConfig config{.bet_sizes = {0.5F, 1.0F}};
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  float initial = evaluator.exploitability(hero, villain);
  for (size_t iteration = 1; iteration <= 1000; ++iteration) {
    evaluator.cfr_iteration(hero, villain, static_cast<float>(iteration));
  }
  float final = evaluator.exploitability(hero, villain);

  check(initial > 0.0F, "multi-size river starts exploitable");
  check(final < initial * 0.25F,
        "multi-size river CFR substantially reduces exploitability");
}

void test_simplified_turn_cfr_reduces_exploitability() {
  std::array<uint8_t, 4> board{0, 5, 10, 15};
  TerminalTables full(board);
  Range hero = first_hands(full.hands(Player::Hero), 4);
  Range villain = last_hands(full.hands(Player::Villain), 4);
  TerminalTables terminals(board, hero, villain);
  GameConfig config{
      .starting_street = StartingStreet::Turn,
      .bet_sizes = {0.5F, 1.0F},
  };
  GameTree tree = build_game_tree(terminals, config);
  RecursiveEvaluator evaluator{tree, terminals};

  float initial = evaluator.exploitability(hero, villain);
  for (size_t iteration = 1; iteration <= 200; ++iteration) {
    evaluator.cfr_iteration(hero, villain, static_cast<float>(iteration));
  }
  float final = evaluator.exploitability(hero, villain);

  check(initial > 0.0F, "simplified turn starts exploitable");
  check(final < initial * 0.1F,
        "simplified turn CFR substantially reduces exploitability");
}

} // namespace

int main() {
  try {
    test_fold_perspective();
    test_showdown_dispatch();
    test_chance_weighting();
    test_player_decision_weighting();
    test_opponent_decision_updates_range();
    test_player_local_ranges();
    test_cfr_update_player_decision();
    test_cfr_update_only_updates_selected_player();
    test_cfr_chance_weights_strategy_sum();
    test_cfr_iteration_updates_both_players();
    test_best_response_selects_best_action();
    test_terminal_game_has_zero_exploitability();
    test_deal_node_defers_blockers_to_terminals();
    test_game_config_builds_trees();
    test_turn_fixture_skips_blocked_decision_hands();
    test_river_cfr_reduces_exploitability();
    test_multi_size_river_cfr_reduces_exploitability();
    test_simplified_turn_cfr_reduces_exploitability();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All recursive evaluator tests passed\n";
  return EXIT_SUCCESS;
}
