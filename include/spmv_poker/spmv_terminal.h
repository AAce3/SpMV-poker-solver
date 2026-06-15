#pragma once

#include "spmv_poker/spmv_evaluator.h"
#include "spmv_poker/terminal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spmv_poker {

constexpr size_t TERMINAL_TILE_SIZE = 8;
constexpr uint16_t INVALID_RANK_GROUP = UINT16_MAX;

/**
 * CPU reference for the intended GPU terminal pipeline:
 *
 *   scatter reaches into rank/card summaries
 *   prefix-scan every summary channel
 *   independently produce each evaluated hand value
 */
struct RankSummaryTerminalOperator final : RiverTerminalOperator {
  RankSummaryTerminalOperator(const TerminalTables &tables,
                              uint64_t flop_mask);
  RankSummaryTerminalOperator(const TerminalTables &tables,
                              uint64_t public_mask,
                              size_t public_card_count);

  [[nodiscard]] std::span<const Hand> hands(Player player) const override;

  [[nodiscard]] size_t showdown_summary_scratch_floats(
      std::span<const BoardIndex> boards, size_t showdown_count) const;
  void reserve_showdown_summary_scratch(std::span<const BoardIndex> boards,
                                        size_t showdown_count) const override;

  void evaluate_folds(BoardIndex board, Player evaluated_player,
                      std::span<const float> opponent_reaches,
                      size_t opponent_hand_stride,
                      std::span<const float> payoffs, std::span<float> values,
                      size_t evaluated_hand_stride) const override;
  void mask_board_reaches(BoardIndex board, Player player,
                          std::span<const float> input,
                          std::span<float> output) const override;
  void mask_board_values(BoardIndex board, Player player,
                         std::span<float> values) const override;
  void evaluate_fold_board_batch(
      std::span<const BoardIndex> boards, Player evaluated_player,
      std::span<const float> opponent_reaches, size_t opponent_hand_stride,
      std::span<const float> payoffs, std::span<float> values,
      size_t evaluated_hand_stride) const override;

  void evaluate_showdowns(BoardIndex board, Player evaluated_player,
                          std::span<const float> opponent_reaches,
                          size_t opponent_hand_stride,
                          std::span<const CompiledShowdown> showdowns,
                          std::span<float> values,
                          size_t evaluated_hand_stride) const override;
  void evaluate_showdown_board_batch(
      std::span<const BoardIndex> boards, Player evaluated_player,
      std::span<const float> opponent_reaches, size_t opponent_hand_stride,
      std::span<const CompiledShowdown> showdowns, std::span<float> values,
      size_t evaluated_hand_stride) const override;

private:
  struct BoardMetadata {
    uint16_t rank_group_count = 0;
    std::array<IndexRange, 2> player_rank_groups;
  };

  [[nodiscard]] size_t runout_for_board(BoardIndex board) const;

  const TerminalTables &tables_;
  std::vector<uint32_t> runout_by_river_board_;
  std::vector<BoardMetadata> board_metadata_;
  std::array<std::vector<uint16_t>, 2> rank_groups_by_board_hand_;
  std::array<std::vector<int32_t>, 2> matching_opponent_hands_;
  std::array<std::vector<float>, 2> board_reach_masks_;
  size_t max_rank_group_count_ = 0;
  mutable std::vector<float> showdown_summary_scratch_;
};

} // namespace spmv_poker
