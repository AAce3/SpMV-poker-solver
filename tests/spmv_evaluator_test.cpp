#include "spmv_poker/spmv_terminal.h"
#include "spmv_poker/spmv_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
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

Range first_hands(std::span<const Hand> hands, size_t count) {
  return {std::vector<Hand>(hands.begin(), hands.begin() + count),
          std::vector<float>(count, 1.0F)};
}

Range last_hands(std::span<const Hand> hands, size_t count) {
  return {std::vector<Hand>(hands.end() - count, hands.end()),
          std::vector<float>(count, 1.0F)};
}

Range masked_range(const Range &range, uint64_t board_mask) {
  Range masked;
  for (size_t index = 0; index < range.hands.size(); ++index) {
    if ((range.hands[index].mask & board_mask) == 0) {
      masked.hands.push_back(range.hands[index]);
      masked.weights.push_back(range.weights[index]);
    }
  }
  return masked;
}

void check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void check_near(float actual, float expected, std::string_view message) {
  if (std::abs(actual - expected) > 1e-6F) {
    throw std::runtime_error(std::string(message));
  }
}

struct TestGame {
  StreetTree game;
  uint32_t villain;
  uint32_t hero;
  size_t fold;
  size_t first_showdown;
  size_t second_showdown;
};

TestGame make_game() {
  StreetTopology topology(Street::River, 3);
  NodeIndex first_showdown_node = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex second_showdown_node = topology.add_showdown_node(2.0F, -2.0F);
  std::array villain_children{first_showdown_node};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain, second_showdown_node};
  NodeIndex hero = topology.add_player_node(Player::Hero, hero_children);
  topology.root = hero;
  uint32_t villain_slot = villain - topology.player_begin();
  uint32_t hero_slot = hero - topology.player_begin();
  size_t fold = topology.nodes[villain].fold_index;
  size_t first_showdown =
      topology.endpoint_for_action(topology.nodes[villain], 1);
  size_t second_showdown =
      topology.endpoint_for_action(topology.nodes[hero], 1);
  StreetTree game(std::move(topology),
                  make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  return {std::move(game), villain_slot,   hero_slot, fold,
          first_showdown,  second_showdown};
}

StreetTopology make_river_topology(size_t hero_hand_count,
                                   size_t villain_hand_count,
                                   float starting_pot,
                                   std::span<const float> bet_sizes) {
  StreetTopology topology(Street::River, hero_hand_count, villain_hand_count);
  assert(starting_pot > 0.0F);
  assert(!bet_sizes.empty());
  float showdown_payoff = 0.5F * starting_pot;

  std::vector<NodeIndex> villain_after_check_children;
  std::vector<NodeIndex> hero_root_children;
  std::vector<NodeIndex> villain_bet_calls;
  std::vector<NodeIndex> hero_bet_calls;
  villain_after_check_children.reserve(bet_sizes.size() + 1);
  hero_root_children.reserve(bet_sizes.size() + 1);
  villain_bet_calls.reserve(bet_sizes.size());
  hero_bet_calls.reserve(bet_sizes.size());

  for (float bet_size : bet_sizes) {
    float called_payoff = showdown_payoff + bet_size * starting_pot;
    hero_bet_calls.push_back(
        topology.add_showdown_node(called_payoff, -called_payoff));
    villain_bet_calls.push_back(
        topology.add_showdown_node(called_payoff, -called_payoff));
  }

  NodeIndex checked_showdown =
      topology.add_showdown_node(showdown_payoff, -showdown_payoff);
  villain_after_check_children.push_back(checked_showdown);

  for (size_t bet = 0; bet < bet_sizes.size(); ++bet) {
    std::array response_children{hero_bet_calls[bet]};
    hero_root_children.push_back(topology.add_player_node(
        Player::Villain, response_children, showdown_payoff));

    std::array checked_response_children{villain_bet_calls[bet]};
    villain_after_check_children.push_back(topology.add_player_node(
        Player::Hero, checked_response_children, -showdown_payoff));
  }

  NodeIndex villain_after_check =
      topology.add_player_node(Player::Villain, villain_after_check_children);
  hero_root_children.insert(hero_root_children.begin(), villain_after_check);
  topology.root = topology.add_player_node(Player::Hero, hero_root_children);
  return topology;
}

StreetTopology make_turn_boundary_topology(size_t hero_hand_count,
                                           size_t villain_hand_count) {
  StreetTopology topology(Street::Turn, hero_hand_count, villain_hand_count);
  NodeIndex showdown = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex boundary = topology.add_boundary_node();
  std::array villain_children{showdown, boundary};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain, showdown};
  topology.root = topology.add_player_node(Player::Hero, hero_children, -2.0F);
  return topology;
}

StreetTopology make_turn_continuation_topology(size_t hero_hand_count,
                                               size_t villain_hand_count) {
  StreetTopology topology(Street::Turn, hero_hand_count, villain_hand_count);
  NodeIndex boundary = topology.add_boundary_node();
  std::array villain_children{boundary};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain};
  topology.root = topology.add_player_node(Player::Hero, hero_children, -2.0F);
  return topology;
}

StreetTopology make_flop_boundary_topology(size_t hero_hand_count,
                                           size_t villain_hand_count) {
  StreetTopology topology(Street::Flop, hero_hand_count, villain_hand_count);
  NodeIndex boundary = topology.add_boundary_node();
  std::array hero_children{boundary};
  topology.root = topology.add_player_node(Player::Hero, hero_children);
  return topology;
}

BoardIndex board_for_mask(const RunoutIndex &boards, uint64_t target_mask) {
  for (BoardIndex board = 0; board < boards.board_count(Street::River);
       ++board) {
    if (boards.board_mask(Street::River, board) == target_mask) {
      return board;
    }
  }
  throw std::runtime_error("missing board target=" +
                           std::to_string(target_mask) +
                           " count=" +
                           std::to_string(boards.board_count(Street::River)));
}

TestGame make_game(const TerminalTables &tables) {
  StreetTopology topology(
      Street::River, tables.hands(Player::Hero).size(),
      tables.hands(Player::Villain).size());
  NodeIndex first_showdown_node = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex second_showdown_node = topology.add_showdown_node(2.0F, -2.0F);
  std::array villain_children{first_showdown_node};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain, second_showdown_node};
  NodeIndex hero = topology.add_player_node(Player::Hero, hero_children);
  topology.root = hero;
  uint32_t villain_slot = villain - topology.player_begin();
  uint32_t hero_slot = hero - topology.player_begin();
  size_t fold = topology.nodes[villain].fold_index;
  size_t first_showdown =
      topology.endpoint_for_action(topology.nodes[villain], 1);
  size_t second_showdown =
      topology.endpoint_for_action(topology.nodes[hero], 1);
  StreetTree game(std::move(topology),
                  make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  return {std::move(game), villain_slot,   hero_slot, fold,
          first_showdown,  second_showdown};
}

void set_regrets(TestGame &test) {
  StreetTree &game = test.game;
  game.regrets[game.state_entry(0, test.hero, 0, 0)] = 3.0F;
  game.regrets[game.state_entry(0, test.hero, 0, 1)] = 1.0F;
  game.regrets[game.state_entry(0, test.hero, 2, 0)] = -1.0F;
  game.regrets[game.state_entry(0, test.hero, 2, 1)] = 4.0F;

  game.regrets[game.state_entry(0, test.villain, 0, 0)] = 1.0F;
  game.regrets[game.state_entry(0, test.villain, 0, 1)] = 3.0F;
}

struct TestTerminalOperator : RiverTerminalOperator {
  mutable bool evaluated_folds = false;
  mutable bool evaluated_showdowns = false;

  std::span<const Hand> hands(Player) const override { return {}; }

  void evaluate_folds(BoardIndex, Player, std::span<const float>, size_t,
                      std::span<const float> payoffs, std::span<float> values,
                      size_t evaluated_hand_stride) const override {
    evaluated_folds = true;
    check(values.size() == payoffs.size() * evaluated_hand_stride,
          "river update provides every fold value");
    std::ranges::fill(values, 3.0F);
  }

  void evaluate_showdowns(BoardIndex, Player, std::span<const float>, size_t,
                          std::span<const CompiledShowdown> showdowns,
                          std::span<float> values,
                          size_t evaluated_hand_stride) const override {
    evaluated_showdowns = true;
    check(values.size() == showdowns.size() * evaluated_hand_stride,
          "river update provides every showdown value");
    for (size_t showdown = 0; showdown < showdowns.size(); ++showdown) {
      std::ranges::fill(values.subspan(showdown * evaluated_hand_stride,
                                       evaluated_hand_stride),
                        10.0F * showdowns[showdown].win_payoff);
    }
  }
};

void test_complete_forward_street_traversal() {
  TestGame test = make_game();
  set_regrets(test);
  size_t width = test.game.compiled.padded_hand_counts[0];

  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][0] = 8.0F;
  roots[0][1] = 6.0F;
  roots[0][2] = 5.0F;
  roots[1][0] = 4.0F;
  roots[1][1] = 2.0F;
  roots[1][2] = 7.0F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};

  StreetReachBuffers reaches;
  propagate_reaches(test.game, 0, root_spans, reaches);
  auto hero_showdowns = showdown_reaches(test.game, reaches, Player::Hero);
  auto hero_folds = fold_reaches(test.game, reaches, Player::Hero);
  auto villain_showdowns =
      showdown_reaches(test.game, reaches, Player::Villain);
  auto villain_folds = fold_reaches(test.game, reaches, Player::Villain);

  check_near(hero_showdowns[test.first_showdown * width], 6.0F,
             "hero reach follows the root strategy through opponent nodes");
  check_near(hero_showdowns[test.second_showdown * width], 2.0F,
             "hero reach reaches the direct showdown action");
  check_near(hero_folds[test.fold * width], 6.0F,
             "hero reach passes unchanged through villain's fold action");

  check_near(villain_showdowns[test.first_showdown * width], 3.0F,
             "villain reach follows villain's continuing strategy");
  check_near(villain_folds[test.fold * width], 1.0F,
             "villain reach follows villain's fold strategy");
  check_near(villain_showdowns[test.second_showdown * width], 4.0F,
             "villain reach passes unchanged through hero's direct action");
}

void compare_batched_forward_against_single_board(
    const TestGame &test, std::span<const BoardIndex> boards,
    std::span<const std::array<std::vector<float>, 2>> root_sets,
    std::span<const std::vector<float>> batched_workspaces) {
  const CompiledStreet &compiled = test.game.compiled;
  check(root_sets.size() == boards.size(),
        "batched forward test provides one root set per board");
  for (size_t board = 0; board < boards.size(); ++board) {
    StreetReachBuffers expected;
    std::array<std::span<const float>, 2> roots{root_sets[board][0],
                                                root_sets[board][1]};
    propagate_reaches(test.game, boards[board], roots, expected);

    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[player_index];
      size_t slot_count =
          compiled.forward_plans[player_index].workspace_slot_count;
      const std::vector<float> &batched = batched_workspaces[player_index];
      for (size_t slot = 0; slot < slot_count; ++slot) {
        auto actual = std::span<const float>(batched)
                          .subspan((slot * boards.size() + board) * width,
                                   width);
        auto expected_slot =
            std::span<const float>(expected.workspaces[player_index])
                .subspan(slot * width, width);
        for (size_t hand = 0; hand < width; ++hand) {
          check_near(actual[hand], expected_slot[hand],
                     "batched forward matches single-board propagation");
        }
      }
    }
  }
}

void test_direct_batched_forward_matches_single_board_and_board_order() {
  TestGame test = make_game();
  set_regrets(test);
  const CompiledStreet &compiled = test.game.compiled;

  std::array<BoardIndex, 3> ordered_boards{0, 17, 42};
  std::array<BoardIndex, 3> shuffled_boards{42, 0, 17};

  std::array<std::array<std::vector<float>, 2>, 3> ordered_roots;
  std::array<std::array<std::vector<float>, 2>, 3> shuffled_roots;

  for (size_t board = 0; board < ordered_boards.size(); ++board) {
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[player_index];
      ordered_roots[board][player_index].resize(width, 0.0F);
      shuffled_roots[board][player_index].resize(width, 0.0F);
      for (size_t hand = 0; hand < test.game.compiled.hand_counts[player_index];
           ++hand) {
        ordered_roots[board][player_index][hand] =
            static_cast<float>(((board + 2) * (player_index + 3) *
                                (hand + 5)) % 37) /
            37.0F;
        shuffled_roots[board][player_index][hand] =
            static_cast<float>(((board + 7) * (player_index + 11) *
                                (hand + 13)) % 41) /
            41.0F;
      }
    }
  }

  std::array<std::vector<float>, 2> ordered_flat_roots{
      std::vector<float>(ordered_boards.size() *
                         compiled.padded_hand_counts[0]),
      std::vector<float>(ordered_boards.size() *
                         compiled.padded_hand_counts[1])};
  std::array<std::vector<float>, 2> shuffled_flat_roots{
      std::vector<float>(shuffled_boards.size() *
                         compiled.padded_hand_counts[0]),
      std::vector<float>(shuffled_boards.size() *
                         compiled.padded_hand_counts[1])};

  for (size_t board = 0; board < ordered_boards.size(); ++board) {
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[player_index];
      auto destination = std::span<float>(ordered_flat_roots[player_index])
                             .subspan(board * width, width);
      std::ranges::copy(ordered_roots[board][player_index], destination.begin());
    }
  }
  for (size_t board = 0; board < shuffled_boards.size(); ++board) {
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[player_index];
      auto destination = std::span<float>(shuffled_flat_roots[player_index])
                             .subspan(board * width, width);
      std::ranges::copy(shuffled_roots[board][player_index],
                        destination.begin());
    }
  }

  auto make_batched = [&](std::span<const BoardIndex> boards,
                          const std::array<std::vector<float>, 2> &flat_roots,
                          std::array<std::vector<float>, 2> &workspaces,
                          std::vector<float> &cumulative_reaches) {
    size_t board_count = boards.size();
    std::array<std::span<float>, 2> output_spans{
        workspaces[0], workspaces[1]};
    cumulative_reaches.assign(board_count, 1.0F);
    StreetBatchView batch{
        .boards = boards,
        .root_reaches = {std::span<const float>(flat_roots[0]),
                         std::span<const float>(flat_roots[1])},
        .root_values = {},
        .cumulative_chance_reaches = cumulative_reaches,
    };
    propagate_reaches_batch(test.game, batch, output_spans);
  };

  std::array<std::vector<float>, 2> ordered_workspace_storage{
      std::vector<float>(compiled.forward_plans[0].workspace_slot_count *
                         ordered_boards.size() *
                         compiled.padded_hand_counts[0]),
      std::vector<float>(compiled.forward_plans[1].workspace_slot_count *
                         ordered_boards.size() *
                         compiled.padded_hand_counts[1])};
  std::vector<float> ordered_cumulative_reaches;
  make_batched(ordered_boards, ordered_flat_roots, ordered_workspace_storage,
               ordered_cumulative_reaches);
  compare_batched_forward_against_single_board(
      test, ordered_boards, ordered_roots, ordered_workspace_storage);

  std::array<std::vector<float>, 2> shuffled_workspace_storage{
      std::vector<float>(compiled.forward_plans[0].workspace_slot_count *
                         shuffled_boards.size() *
                         compiled.padded_hand_counts[0]),
      std::vector<float>(compiled.forward_plans[1].workspace_slot_count *
                         shuffled_boards.size() *
                         compiled.padded_hand_counts[1])};
  std::vector<float> shuffled_cumulative_reaches;
  std::array<std::array<std::vector<float>, 2>, 3> shuffled_expected;
  for (size_t index = 0; index < shuffled_boards.size(); ++index) {
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[player_index];
      shuffled_expected[index][player_index].resize(width, 0.0F);
      for (size_t hand = 0; hand < test.game.compiled.hand_counts[player_index];
           ++hand) {
        shuffled_expected[index][player_index][hand] =
            static_cast<float>(((index + 11) * (player_index + 5) *
                                (hand + 7)) % 43) /
            43.0F;
      }
    }
  }
  make_batched(shuffled_boards, shuffled_flat_roots, shuffled_workspace_storage,
               shuffled_cumulative_reaches);
  compare_batched_forward_against_single_board(
      test, shuffled_boards, shuffled_expected, shuffled_workspace_storage);
}

void test_complete_backward_street_update() {
  TestGame test = make_game();
  set_regrets(test);
  size_t width = test.game.compiled.padded_hand_counts[0];

  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][0] = 2.0F;
  roots[0][1] = 3.0F;
  roots[0][2] = 0.5F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};
  StreetReachBuffers reaches;
  propagate_reaches(test.game, 0, root_spans, reaches);

  StreetValueBuffer values;
  prepare_value_buffer(test.game, Player::Hero, values);
  auto showdowns = showdown_values(test.game, values, Player::Hero);
  auto folds = fold_values(test.game, values, Player::Hero);
  showdowns[test.first_showdown * width + 0] = 2.0F;
  showdowns[test.first_showdown * width + 1] = -2.0F;
  showdowns[test.first_showdown * width + 2] = 5.0F;
  showdowns[test.second_showdown * width + 0] = 6.0F;
  showdowns[test.second_showdown * width + 1] = 4.0F;
  showdowns[test.second_showdown * width + 2] = -1.0F;
  folds[test.fold * width + 0] = 1.0F;
  folds[test.fold * width + 1] = 1.0F;
  folds[test.fold * width + 2] = 1.0F;

  std::vector<float> root_values(width);
  update_player(test.game, 0, Player::Hero, reaches.workspaces[0], values,
                root_values, 0.5F, 2.0F);

  check_near(root_values[0], 3.75F,
             "root mixes the opponent-node sum and direct action");
  check_near(root_values[1], 1.5F,
             "uniform root strategy mixes complete child values");
  check_near(root_values[2], -1.0F,
             "positive regret selects the direct action");
  check_near(
      test.game.cumulative_strategy[test.game.state_entry(0, test.hero, 0, 0)],
      3.0F, "whole-street update accumulates root strategy");
  check_near(
      test.game
          .cumulative_strategy[test.game.state_entry(0, test.villain, 0, 0)],
      0.0F, "hero update leaves villain strategy state unchanged");
}

void test_multiple_hand_blocks() {
  StreetTopology topology(Street::River, 35);
  NodeIndex first = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex second = topology.add_showdown_node(2.0F, -2.0F);
  std::array children{first, second};
  topology.root = topology.add_player_node(Player::Hero, children);
  size_t first_showdown =
      topology.endpoint_for_action(topology.nodes[topology.root], 0);
  size_t second_showdown =
      topology.endpoint_for_action(topology.nodes[topology.root], 1);
  StreetTree game(std::move(topology),
                  make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  size_t width = game.compiled.padded_hand_counts[0];

  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][34] = 8.0F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};
  StreetReachBuffers reaches;
  propagate_reaches(game, 0, root_spans, reaches);

  auto reached_showdowns = showdown_reaches(game, reaches, Player::Hero);
  check_near(reached_showdowns[first_showdown * width + 34], 4.0F,
             "second hand block propagates its first action");
  check_near(reached_showdowns[second_showdown * width + 34], 4.0F,
             "second hand block propagates its second action");

  StreetValueBuffer values;
  prepare_value_buffer(game, Player::Hero, values);
  auto showdowns = showdown_values(game, values, Player::Hero);
  showdowns[first_showdown * width + 34] = 2.0F;
  showdowns[second_showdown * width + 34] = 6.0F;
  std::vector<float> root_values(width);
  update_player(game, 0, Player::Hero, reaches.workspaces[0], values,
                root_values);
  check_near(root_values[34], 4.0F,
             "second hand block reduces complete street values");
}

void test_player_specific_workspace_widths() {
  StreetTopology topology(Street::River, 3, 35);
  NodeIndex showdown = topology.add_showdown_node(1.0F, -1.0F);
  std::array children{showdown};
  topology.root = topology.add_player_node(Player::Hero, children);
  StreetTree game(std::move(topology),
                  make_mask(std::array<uint8_t, 3>{2, 5, 8}));

  std::array<std::vector<float>, 2> roots{
      std::vector<float>(game.compiled.padded_hand_counts[0]),
      std::vector<float>(game.compiled.padded_hand_counts[1]),
  };
  roots[0][2] = 3.0F;
  roots[1][34] = 7.0F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};
  StreetReachBuffers reaches;
  propagate_reaches(game, 0, root_spans, reaches);

  auto hero_showdowns = showdown_reaches(game, reaches, Player::Hero);
  auto villain_showdowns = showdown_reaches(game, reaches, Player::Villain);
  check(hero_showdowns.size() == game.compiled.padded_hand_counts[0],
        "hero showdown region uses hero width");
  check(villain_showdowns.size() == game.compiled.padded_hand_counts[1],
        "villain showdown region uses villain width");
  check_near(hero_showdowns[2], 3.0F, "hero workspace preserves hero hand");
  check_near(villain_showdowns[34], 7.0F,
             "villain workspace preserves villain hand");
}

void test_separate_showdown_and_boundary_regions() {
  StreetTopology topology(Street::Turn, 3);
  NodeIndex showdown = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex boundary = topology.add_boundary_node();
  std::array children{showdown, boundary};
  topology.root = topology.add_player_node(Player::Hero, children);
  StreetTree game(std::move(topology),
                  make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  size_t width = game.compiled.padded_hand_counts[0];

  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][0] = 8.0F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};
  StreetReachBuffers reaches;
  propagate_reaches(game, 0, root_spans, reaches);

  auto reached_showdowns = showdown_reaches(game, reaches, Player::Hero);
  auto reached_boundaries = boundary_reaches(game, reaches, Player::Hero);
  check(reached_showdowns.size() == width,
        "showdown reaches occupy their own region");
  check(reached_boundaries.size() == width,
        "boundary reaches occupy their own region");
  check_near(reached_showdowns[0], 4.0F,
             "showdown action writes to showdown region");
  check_near(reached_boundaries[0], 4.0F,
             "boundary action writes to boundary region");

  StreetValueBuffer values;
  prepare_value_buffer(game, Player::Hero, values);
  showdown_values(game, values, Player::Hero)[0] = 2.0F;
  boundary_values(game, values, Player::Hero)[0] = 6.0F;
  std::vector<float> root_values(width);
  update_player(game, 0, Player::Hero, reaches.workspaces[0], values,
                root_values);
  check_near(root_values[0], 4.0F,
             "backward pass reads separated terminal regions");
}

void test_river_update_orchestration() {
  TestGame test = make_game();
  size_t width = test.game.compiled.padded_hand_counts[0];
  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][0] = 1.0F;
  roots[1][0] = 1.0F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};

  TestTerminalOperator terminals;
  RiverUpdateBuffers buffers;
  std::vector<float> root_values(width);
  update_river(test.game, 0, root_spans, Player::Hero, terminals, buffers,
               root_values);

  check(terminals.evaluated_folds, "river update evaluates folds");
  check(terminals.evaluated_showdowns, "river update evaluates showdowns");
  check_near(root_values[0], 16.5F,
             "river update propagates terminal values to the root");
}

void test_cpu_solve_executor_matches_river_update() {
  TestGame expected = make_game();
  TestGame actual = make_game();
  set_regrets(expected);
  set_regrets(actual);

  size_t width = expected.game.compiled.padded_hand_counts[0];
  std::array<std::vector<float>, 2> roots{std::vector<float>(width),
                                          std::vector<float>(width)};
  roots[0][0] = 2.0F;
  roots[0][1] = 3.0F;
  roots[0][2] = 0.5F;
  std::array<std::span<const float>, 2> root_spans{roots[0], roots[1]};

  TestTerminalOperator terminals;
  RiverUpdateBuffers expected_buffers;
  std::vector<float> expected_root_values(width);
  update_river(expected.game, 0, root_spans, Player::Hero, terminals,
               expected_buffers, expected_root_values, 1.0F, 2.0F);

  SolveProgram program{
      .board_index = actual.game.board_index,
      .river = actual.game.compiled,
      .schedule = ExecutionSchedule{},
      .terminals = &terminals,
  };
  SolveState state{
      .river = {.regrets = actual.game.regrets,
                .cumulative_strategy = actual.game.cumulative_strategy},
  };
  CpuSolveWorkspace workspace;
  workspace.river.prepare(program.river, 1);
  CpuSolveExecutor executor;
  std::vector<float> actual_root_values(width);
  executor.update_player(program, state, workspace, Player::Hero, root_spans,
                         actual_root_values, 2.0F);

  StreetTree solved_tree(program.river, *program.board_index,
                         std::move(state.river.regrets),
                         std::move(state.river.cumulative_strategy));

  for (size_t hand = 0; hand < width; ++hand) {
    check_near(actual_root_values[hand], expected_root_values[hand],
               "executor root values match river update");
  }
  for (size_t entry = 0; entry < expected.game.regrets.size(); ++entry) {
    check_near(solved_tree.regrets[entry], expected.game.regrets[entry],
               "executor regret update matches river update");
    check_near(solved_tree.cumulative_strategy[entry],
               expected.game.cumulative_strategy[entry],
               "executor strategy update matches river update");
  }
}

} // namespace

int main() {
  try {
    test_complete_forward_street_traversal();
    test_complete_backward_street_update();
    test_multiple_hand_blocks();
    test_player_specific_workspace_widths();
    test_separate_showdown_and_boundary_regions();
    test_river_update_orchestration();
    test_cpu_solve_executor_matches_river_update();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All SpMV evaluator tests passed\n";
  return EXIT_SUCCESS;
}
