#include "spmv_poker/spmv_terminal.h"

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
  if (std::abs(actual - expected) > 0.003F) {
    throw std::runtime_error(std::string(message));
  }
}

Range selected_range(std::span<const Hand> hands, size_t divisor,
                     size_t remainder) {
  Range range;
  for (size_t hand = 0; hand < hands.size(); ++hand) {
    if (hand % divisor == remainder) {
      range.hands.push_back(hands[hand]);
      range.weights.push_back(1.0F);
    }
  }
  return range;
}

size_t runout_for_board(const TerminalTables &tables, const RunoutIndex &boards,
                        BoardIndex board) {
  uint64_t board_mask = boards.board_mask(Street::River, board);
  for (size_t runout = 0; runout < tables.runouts.size(); ++runout) {
    if (make_mask(tables.runouts[runout]) == board_mask) {
      return runout;
    }
  }
  throw std::runtime_error("missing runout");
}

void test_tiled_rank_scan_matches_terminal_tables() {
  std::array<uint8_t, 3> flop{0, 5, 10};
  TerminalTables full(flop);
  Range hero = selected_range(full.hands(Player::Hero), 2, 0);
  Range villain = selected_range(full.hands(Player::Villain), 3, 1);
  TerminalTables tables(flop, hero, villain);
  RankSummaryTerminalOperator terminals(tables, make_mask(flop));
  RunoutIndex boards(make_mask(flop));

  constexpr size_t terminal_count = TERMINAL_TILE_SIZE + 3;
  size_t opponent_stride = villain.hands.size() + 5;
  size_t evaluated_stride = hero.hands.size() + 7;
  std::vector<float> reaches(terminal_count * opponent_stride);
  for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
    for (size_t hand = 0; hand < villain.hands.size(); ++hand) {
      reaches[terminal * opponent_stride + hand] =
          static_cast<float>(((terminal + 2) * (hand + 3)) % 17) / 17.0F;
    }
  }

  std::vector<CompiledShowdown> showdowns;
  std::vector<float> fold_payoffs;
  for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
    float scale = static_cast<float>(terminal + 1) / 3.0F;
    showdowns.push_back({.win_payoff = scale, .loss_payoff = -2.0F * scale});
    fold_payoffs.push_back(scale);
  }

  for (BoardIndex board : {BoardIndex{0}, BoardIndex{317}, BoardIndex{2351}}) {
    size_t runout = runout_for_board(tables, boards, board);
    std::vector<float> actual_showdown(terminal_count * evaluated_stride);
    terminals.evaluate_showdowns(board, Player::Hero, reaches, opponent_stride,
                                 showdowns, actual_showdown, evaluated_stride);

    std::vector<float> actual_fold(terminal_count * evaluated_stride);
    terminals.evaluate_folds(board, Player::Hero, reaches, opponent_stride,
                             fold_payoffs, actual_fold, evaluated_stride);

    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      std::span<const float> reach =
          std::span<const float>(reaches)
              .subspan(terminal * opponent_stride, villain.hands.size());
      std::vector<float> expected_showdown(hero.hands.size());
      tables.apply_showdown(Player::Hero, runout, reach,
                            showdowns[terminal].win_payoff,
                            showdowns[terminal].loss_payoff,
                            expected_showdown);
      std::vector<float> expected_fold(hero.hands.size());
      tables.apply_fold(Player::Hero, reach, fold_payoffs[terminal],
                        expected_fold);
      for (size_t hand = 0; hand < hero.hands.size(); ++hand) {
        check_close(actual_showdown[terminal * evaluated_stride + hand],
                    expected_showdown[hand], "tiled showdown mismatch");
        check_close(actual_fold[terminal * evaluated_stride + hand],
                    expected_fold[hand], "tiled fold mismatch");
      }
      for (size_t hand = hero.hands.size(); hand < evaluated_stride; ++hand) {
        check(actual_showdown[terminal * evaluated_stride + hand] == 0.0F,
              "showdown clears padded output");
        check(actual_fold[terminal * evaluated_stride + hand] == 0.0F,
              "fold clears padded output");
      }
    }
  }
}

void test_both_evaluated_players() {
  std::array<uint8_t, 3> flop{0, 5, 10};
  TerminalTables tables(flop);
  RankSummaryTerminalOperator terminals(tables, make_mask(flop));
  size_t stride = tables.hands(Player::Hero).size();
  std::vector<float> reaches(stride, 1.0F);
  std::array showdowns{CompiledShowdown{.win_payoff = 1.0F,
                                        .loss_payoff = -1.0F}};
  std::vector<float> values(stride);
  terminals.evaluate_showdowns(1175, Player::Villain, reaches, stride,
                               showdowns, values, stride);

  RunoutIndex boards(make_mask(flop));
  size_t runout = runout_for_board(tables, boards, 1175);
  std::vector<float> expected(stride);
  tables.apply_showdown(Player::Villain, runout, reaches, 1.0F, -1.0F,
                        expected);
  for (size_t hand = 0; hand < stride; ++hand) {
    check_close(values[hand], expected[hand], "villain showdown mismatch");
  }
}

void test_board_batch_layout_matches_individual_evaluation() {
  std::array<uint8_t, 3> flop{0, 5, 10};
  TerminalTables full(flop);
  Range hero = selected_range(full.hands(Player::Hero), 2, 0);
  Range villain = selected_range(full.hands(Player::Villain), 3, 1);
  TerminalTables tables(flop, hero, villain);
  RankSummaryTerminalOperator terminals(tables, make_mask(flop));

  std::array<BoardIndex, 3> boards{0, 317, 2351};
  constexpr size_t terminal_count = 3;
  size_t opponent_stride = villain.hands.size() + 5;
  size_t evaluated_stride = hero.hands.size() + 7;
  std::vector<float> reaches(terminal_count * boards.size() * opponent_stride);
  for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
    for (size_t board = 0; board < boards.size(); ++board) {
      for (size_t hand = 0; hand < villain.hands.size(); ++hand) {
        reaches[(terminal * boards.size() + board) * opponent_stride + hand] =
            static_cast<float>(((terminal + 2) * (board + 3) * (hand + 5)) %
                               19) /
            19.0F;
      }
    }
  }

  std::array showdowns{
      CompiledShowdown{.win_payoff = 1.0F, .loss_payoff = -1.0F},
      CompiledShowdown{.win_payoff = 2.0F, .loss_payoff = -3.0F},
      CompiledShowdown{.win_payoff = 4.0F, .loss_payoff = -2.0F},
  };
  std::array fold_payoffs{1.0F, 2.0F, 3.0F};
  std::vector<float> showdown_values(terminal_count * boards.size() *
                                     evaluated_stride);
  std::vector<float> fold_values(terminal_count * boards.size() *
                                 evaluated_stride);
  terminals.evaluate_showdown_board_batch(
      boards, Player::Hero, reaches, opponent_stride, showdowns,
      showdown_values, evaluated_stride);
  terminals.evaluate_fold_board_batch(boards, Player::Hero, reaches,
                                      opponent_stride, fold_payoffs,
                                      fold_values, evaluated_stride);

  for (size_t board = 0; board < boards.size(); ++board) {
    std::vector<float> board_reaches(terminal_count * opponent_stride);
    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      auto source = std::span<const float>(reaches).subspan(
          (terminal * boards.size() + board) * opponent_stride,
          opponent_stride);
      std::ranges::copy(source, board_reaches.begin() +
                                   terminal * opponent_stride);
    }
    std::vector<float> expected_showdowns(terminal_count * evaluated_stride);
    std::vector<float> expected_folds(terminal_count * evaluated_stride);
    terminals.evaluate_showdowns(boards[board], Player::Hero, board_reaches,
                                 opponent_stride, showdowns, expected_showdowns,
                                 evaluated_stride);
    terminals.evaluate_folds(boards[board], Player::Hero, board_reaches,
                             opponent_stride, fold_payoffs, expected_folds,
                             evaluated_stride);
    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      for (size_t hand = 0; hand < evaluated_stride; ++hand) {
        size_t batched =
            (terminal * boards.size() + board) * evaluated_stride + hand;
        size_t expected = terminal * evaluated_stride + hand;
        check_close(showdown_values[batched], expected_showdowns[expected],
                    "batched showdown layout mismatch");
        check_close(fold_values[batched], expected_folds[expected],
                    "batched fold layout mismatch");
      }
    }
  }
}

void test_native_board_batch_matches_individual_evaluation() {
  std::array<uint8_t, 3> flop{0, 5, 10};
  TerminalTables full(flop);
  Range hero = selected_range(full.hands(Player::Hero), 2, 0);
  Range villain = selected_range(full.hands(Player::Villain), 3, 1);
  TerminalTables tables(flop, hero, villain);
  RankSummaryTerminalOperator terminals(tables, make_mask(flop));

  auto check_player = [&](Player evaluated_player,
                          std::span<const BoardIndex> boards,
                          size_t terminal_count, float player_scale) {
    auto evaluated_hands = tables.hands(evaluated_player);
    auto opponent_hands = tables.hands(opponent(evaluated_player));
    size_t opponent_stride = opponent_hands.size() + 5;
    size_t evaluated_stride = evaluated_hands.size() + 7;
    std::vector<float> reaches(terminal_count * boards.size() *
                               opponent_stride);
    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      for (size_t board = 0; board < boards.size(); ++board) {
        for (size_t hand = 0; hand < opponent_hands.size(); ++hand) {
          reaches[(terminal * boards.size() + board) * opponent_stride + hand] =
              static_cast<float>(((terminal + 2) * (board + 3) *
                                  (hand + 5) * static_cast<size_t>(player_scale * 10.0F)) %
                                 23) /
              23.0F;
        }
      }
    }

    std::vector<CompiledShowdown> showdowns;
    std::vector<float> fold_payoffs;
    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      float scale = player_scale * static_cast<float>(terminal + 1) / 4.0F;
      showdowns.push_back(
          {.win_payoff = scale, .loss_payoff = -1.5F * scale});
      fold_payoffs.push_back(0.5F * scale);
    }

    std::vector<float> showdown_values(terminal_count * boards.size() *
                                       evaluated_stride);
    std::vector<float> fold_values(terminal_count * boards.size() *
                                   evaluated_stride);
    terminals.evaluate_showdown_board_batch(
        boards, evaluated_player, reaches, opponent_stride, showdowns,
        showdown_values, evaluated_stride);
    terminals.evaluate_fold_board_batch(boards, evaluated_player, reaches,
                                        opponent_stride, fold_payoffs,
                                        fold_values, evaluated_stride);

    for (size_t board = 0; board < boards.size(); ++board) {
      std::vector<float> board_reaches(terminal_count * opponent_stride);
      for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
        auto source = std::span<const float>(reaches).subspan(
            (terminal * boards.size() + board) * opponent_stride,
            opponent_stride);
        std::ranges::copy(source,
                          board_reaches.begin() + terminal * opponent_stride);
      }

      std::vector<float> expected_showdowns(terminal_count * evaluated_stride);
      std::vector<float> expected_folds(terminal_count * evaluated_stride);
      terminals.evaluate_showdowns(boards[board], evaluated_player,
                                   board_reaches, opponent_stride, showdowns,
                                   expected_showdowns, evaluated_stride);
      terminals.evaluate_folds(boards[board], evaluated_player, board_reaches,
                               opponent_stride, fold_payoffs, expected_folds,
                               evaluated_stride);

      for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
        for (size_t hand = 0; hand < evaluated_stride; ++hand) {
          size_t batched =
              (terminal * boards.size() + board) * evaluated_stride + hand;
          size_t expected = terminal * evaluated_stride + hand;
          check_close(showdown_values[batched], expected_showdowns[expected],
                      "native batched showdown mismatch");
          check_close(fold_values[batched], expected_folds[expected],
                      "native batched fold mismatch");
        }
      }
    }
  };

  std::array<BoardIndex, 4> first_boards{0, 317, 761, 1175};
  std::array<BoardIndex, 5> second_boards{1, 88, 430, 901, 1111};
  check_player(Player::Hero, first_boards, TERMINAL_TILE_SIZE + 5, 1.0F);
  check_player(Player::Villain, first_boards, TERMINAL_TILE_SIZE + 5,
               1.25F);
  check_player(Player::Hero, second_boards, 4, 1.5F);
  check_player(Player::Villain, second_boards, 4, 1.75F);
}

} // namespace

int main() {
  try {
    test_tiled_rank_scan_matches_terminal_tables();
    test_both_evaluated_players();
    test_board_batch_layout_matches_individual_evaluation();
    test_native_board_batch_matches_individual_evaluation();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "All rank summary terminal tests passed\n";
  return EXIT_SUCCESS;
}
