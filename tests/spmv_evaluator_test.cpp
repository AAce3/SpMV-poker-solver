#include "spmv_poker/spmv_terminal.h"
#include "spmv_poker/spmv_evaluator.h"

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

void test_dense_river_board_batch_matches_individual_updates() {
  TestGame individual = make_game();
  TestGame batched = make_game();
  size_t width = individual.game.compiled.padded_hand_counts[0];
  constexpr size_t board_count = 5;
  std::array<std::array<std::vector<float>, 2>, board_count> roots;
  std::array<std::vector<float>, board_count> individual_values;
  std::array<std::vector<float>, board_count> batched_values;
  std::vector<RiverBoardUpdate> jobs;
  jobs.reserve(board_count);
  for (size_t board = 0; board < board_count; ++board) {
    roots[board][0].resize(width);
    roots[board][1].resize(width);
    roots[board][0][0] = static_cast<float>(board + 1);
    roots[board][1][0] = static_cast<float>(2 * board + 1);
    individual_values[board].resize(width);
    batched_values[board].resize(width);
    jobs.push_back({
        .board = static_cast<BoardIndex>(board),
        .root_reaches = {roots[board][0], roots[board][1]},
        .root_values = batched_values[board],
        .chance_reach = 0.25F * static_cast<float>(board + 1),
    });
  }
  TestTerminalOperator terminals;
  RiverUpdateBuffers individual_buffers;
  for (size_t board = 0; board < board_count; ++board) {
    std::array<std::span<const float>, 2> root_spans{roots[board][0],
                                                     roots[board][1]};
    update_river(individual.game, static_cast<BoardIndex>(board), root_spans,
                 Player::Hero, terminals, individual_buffers,
                 individual_values[board],
                 0.25F * static_cast<float>(board + 1), 2.0F);
  }

  RiverBoardBatchBuffers batch_buffers;
  update_river_boards(batched.game, jobs, Player::Hero, terminals,
                      batch_buffers, 2.0F);

  for (size_t board = 0; board < board_count; ++board) {
    for (size_t hand = 0; hand < width; ++hand) {
      check_near(batched_values[board][hand], individual_values[board][hand],
                 "board batch root value matches individual update");
    }
  }
  for (size_t entry = 0; entry < individual.game.regrets.size(); ++entry) {
    check_near(batched.game.regrets[entry], individual.game.regrets[entry],
               "board batch regret update matches individual update");
    check_near(batched.game.cumulative_strategy[entry],
               individual.game.cumulative_strategy[entry],
               "board batch strategy update matches individual update");
  }
  size_t expected_reach_floats_per_board =
      static_cast<size_t>(
          batched.game.compiled.forward_plans[0].workspace_slot_count) *
          width +
      static_cast<size_t>(
          batched.game.compiled.forward_plans[1].workspace_slot_count) *
          width;
  check(batch_buffers.reaches[0].size() + batch_buffers.reaches[1].size() ==
            board_count * expected_reach_floats_per_board,
        "board batch reach workspace has a dense board dimension");
  check(batch_buffers.values.size() ==
            board_count *
                batched.game.compiled.value_workspace_slot_count * width,
        "board batch value workspace has a dense board dimension");
}

void test_rank_summary_terminal_board_batch_integration() {
  std::array<uint8_t, 3> flop{2, 5, 8};
  TerminalTables tables(flop);
  TestGame individual = make_game(tables);
  TestGame batched = make_game(tables);
  TiledRankSummaryRiverTerminalOperator terminals(tables, make_mask(flop));

  size_t width = individual.game.compiled.padded_hand_counts[0];
  constexpr size_t board_count = 3;
  std::array<BoardIndex, board_count> boards{0, 317, 761};
  std::array<std::array<std::vector<float>, 2>, board_count> roots;
  std::array<std::vector<float>, board_count> individual_values;
  std::array<std::vector<float>, board_count> batched_values;
  std::vector<RiverBoardUpdate> jobs;
  jobs.reserve(board_count);
  for (size_t board = 0; board < board_count; ++board) {
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      roots[board][player_index].resize(width);
      for (size_t hand = 0; hand < width; ++hand) {
        roots[board][player_index][hand] =
            static_cast<float>(((board + 2) * (player_index + 3) *
                                (hand + 5)) % 29) /
            29.0F;
      }
    }
    individual_values[board].resize(width);
    batched_values[board].resize(width);
    jobs.push_back({
        .board = boards[board],
        .root_reaches = {roots[board][0], roots[board][1]},
        .root_values = batched_values[board],
        .chance_reach = 0.25F * static_cast<float>(board + 1),
    });
  }

  RiverUpdateBuffers individual_buffers;
  for (size_t board = 0; board < board_count; ++board) {
    std::array<std::span<const float>, 2> root_spans{roots[board][0],
                                                     roots[board][1]};
    update_river(individual.game, boards[board], root_spans, Player::Hero,
                 terminals, individual_buffers, individual_values[board],
                 0.25F * static_cast<float>(board + 1), 1.5F);
  }

  RiverBoardBatchBuffers batch_buffers;
  update_river_boards(batched.game, jobs, Player::Hero, terminals,
                      batch_buffers, 1.5F);

  for (size_t board = 0; board < board_count; ++board) {
    for (size_t hand = 0; hand < width; ++hand) {
      check_near(batched_values[board][hand], individual_values[board][hand],
                 "rank-summary batched root value matches individual update");
    }
  }
  for (size_t entry = 0; entry < individual.game.regrets.size(); ++entry) {
    check_near(batched.game.regrets[entry], individual.game.regrets[entry],
               "rank-summary batched regret update matches individual update");
    check_near(batched.game.cumulative_strategy[entry],
               individual.game.cumulative_strategy[entry],
               "rank-summary batched strategy update matches individual update");
  }

  size_t expected_reach_floats_per_board =
      static_cast<size_t>(
          batched.game.compiled.forward_plans[0].workspace_slot_count) *
          width +
      static_cast<size_t>(
          batched.game.compiled.forward_plans[1].workspace_slot_count) *
          width;
  check(batch_buffers.reaches[0].size() + batch_buffers.reaches[1].size() ==
            board_count * expected_reach_floats_per_board,
        "rank-summary board batch reach workspace has a dense board dimension");
  check(batch_buffers.values.size() ==
            board_count *
                batched.game.compiled.value_workspace_slot_count * width,
        "rank-summary board batch value workspace has a dense board dimension");
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
    test_dense_river_board_batch_matches_individual_updates();
    test_rank_summary_terminal_board_batch_integration();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All SpMV evaluator tests passed\n";
  return EXIT_SUCCESS;
}
