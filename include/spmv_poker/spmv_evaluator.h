#pragma once

#include "spmv_poker/tree_compiler.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace spmv_poker {

namespace detail {

using HandBlock = std::span<float, HAND_BLOCK_SIZE>;
using ConstHandBlock = std::span<const float, HAND_BLOCK_SIZE>;

inline HandBlock batched_hand_block_at_slot(std::span<float> storage,
                                            uint32_t slot, size_t board,
                                            size_t board_count,
                                            size_t hands_per_slot,
                                            size_t hand_begin) {
  return HandBlock(
      storage.data() +
          (static_cast<size_t>(slot) * board_count + board) * hands_per_slot +
          hand_begin,
      HAND_BLOCK_SIZE);
}

inline ConstHandBlock batched_hand_block_at_slot(
    std::span<const float> storage, uint32_t slot, size_t board,
    size_t board_count, size_t hands_per_slot, size_t hand_begin) {
  return ConstHandBlock(
      storage.data() +
          (static_cast<size_t>(slot) * board_count + board) * hands_per_slot +
          hand_begin,
      HAND_BLOCK_SIZE);
}

template <typename Action>
inline void compute_regret_matching_factors(
    std::span<const float> board_regrets, uint32_t cfr_offset_within_board,
    uint32_t action_begin, uint16_t action_count, size_t hand_begin,
    std::span<const Action> actions, HandBlock inverse_positive_sums) {
  std::ranges::fill(inverse_positive_sums, 0.0F);
  for (size_t action = 0; action < action_count; ++action) {
    size_t action_offset =
        actions[action_begin + action].cfr_action_offset + hand_begin;
    const float *regrets =
        board_regrets.data() + cfr_offset_within_board + action_offset;
    for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
      inverse_positive_sums[lane] += std::max(0.0F, regrets[lane]);
    }
  }

  for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
    float positive_sum = inverse_positive_sums[lane];
    inverse_positive_sums[lane] =
        positive_sum > 0.0F ? 1.0F / positive_sum : 0.0F;
  }
}

constexpr Street next_street(Street street) {
  switch (street) {
  case Street::Flop:
    return Street::Turn;
  case Street::Turn:
    return Street::River;
  case Street::River:
    return Street::River;
  }
  return Street::River;
}

} // namespace detail

struct StreetReachBuffers {
  std::array<std::vector<float>, 2> workspaces;
};

struct StreetValueBuffer {
  std::vector<float> workspace;
};

struct RiverUpdateBuffers {
  StreetReachBuffers reaches;
  StreetValueBuffer values;
};

struct ExecutionCapacities {
  size_t flop_boards = 0;
  size_t turn_boards = 0;
  size_t river_boards = 0;
  size_t flop_to_turn_transition_boards = 0;
  size_t turn_to_river_transition_boards = 0;
};

using RootRangesView = std::array<std::span<const float>, 2>;

struct RiverTerminalOperator;

struct CpuBoundaryValueAccumulator {
  std::array<std::vector<float>, 2> values;
  size_t endpoint_capacity = 0;
  size_t parent_board_capacity = 0;

  void prepare(size_t endpoint_count, size_t parent_board_count,
               const CompiledStreet &compiled) {
    endpoint_capacity = endpoint_count;
    parent_board_capacity = parent_board_count;
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[index];
      values[index].resize(endpoint_capacity * parent_board_capacity * width);
    }
  }
};

struct CpuTransitionWorkspace {
  std::array<std::vector<float>, 2> child_root_reaches;
  std::array<std::vector<float>, 2> child_root_values;
  std::vector<BoardIndex> child_boards;
  std::vector<uint32_t> parent_board_for_child;
  std::vector<float> local_chance_weights;
  std::vector<float> child_cumulative_chance_reaches;
  size_t endpoint_capacity = 0;
  size_t parent_board_capacity = 0;
  size_t child_board_capacity = 0;

  void prepare(size_t endpoint_count, size_t parent_board_count,
               size_t child_board_count, const CompiledStreet &child_compiled) {
    endpoint_capacity = endpoint_count;
    parent_board_capacity = parent_board_count;
    child_board_capacity = child_board_count;
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t index = static_cast<size_t>(player);
      size_t width = child_compiled.padded_hand_counts[index];
      child_root_reaches[index].resize(endpoint_capacity * child_board_capacity *
                                       width);
      child_root_values[index].resize(endpoint_capacity * child_board_capacity *
                                      width);
    }
    child_boards.resize(child_board_capacity);
    parent_board_for_child.resize(child_board_capacity);
    local_chance_weights.resize(child_board_capacity);
    child_cumulative_chance_reaches.resize(child_board_capacity);
  }
};

struct CpuStreetWorkspace {
  std::array<std::vector<float>, 2> reaches;
  std::array<std::vector<float>, 2> values;
  size_t board_capacity = 0;

  void prepare(const CompiledStreet &compiled, size_t requested_board_capacity) {
    board_capacity = requested_board_capacity;
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t index = static_cast<size_t>(player);
      size_t width = compiled.padded_hand_counts[index];
      size_t reach_slots = compiled.forward_plans[index].workspace_slot_count;
      reaches[index].resize(reach_slots * board_capacity * width);
      values[index].resize(compiled.value_workspace_slot_count * board_capacity *
                           width);
    }
  }
};

struct SolveProgram {
  std::optional<RunoutIndex> board_index;
  std::optional<CompiledStreet> flop;
  std::optional<CompiledStreet> turn;
  CompiledStreet river;
  std::optional<CompiledTransitionGraph> flop_transition_graph;
  std::optional<CompiledTransitionGraph> turn_transition_graph;
  ExecutionSchedule schedule;
  const RiverTerminalOperator *terminals = nullptr;
};

struct StreetState {
  std::vector<float> regrets;
  std::vector<float> cumulative_strategy;
};

struct SolveState {
  std::optional<StreetState> flop;
  std::optional<StreetState> turn;
  StreetState river;
};

struct CpuSolveWorkspace {
  std::optional<CpuStreetWorkspace> flop;
  std::optional<CpuStreetWorkspace> turn;
  CpuStreetWorkspace river;
  std::optional<CpuTransitionWorkspace> flop_to_turn;
  std::optional<CpuTransitionWorkspace> turn_to_river;
  std::optional<CpuBoundaryValueAccumulator> flop_boundary_values;
  std::optional<CpuBoundaryValueAccumulator> turn_boundary_values;
  CpuSolveWorkspace() = default;
};

class CpuSolveExecutor {
public:
  void update_player(const SolveProgram &program, SolveState &state,
                     CpuSolveWorkspace &workspace, Player player,
                     RootRangesView root_ranges, std::span<float> root_values,
                     float iteration_weight);
};

template <typename T>
class BoardHandView {
public:
  BoardHandView(std::span<T> storage, size_t board_count, size_t hand_stride)
      : storage_(storage), board_count_(board_count),
        hand_stride_(hand_stride) {
    assert(storage_.size() >= board_count_ * hand_stride_);
  }

  [[nodiscard]] T &operator()(size_t board, size_t hand) const {
    assert(board < board_count_);
    assert(hand < hand_stride_);
    return storage_[board * hand_stride_ + hand];
  }

  [[nodiscard]] std::span<T> hands(size_t board) const {
    assert(board < board_count_);
    return storage_.subspan(board * hand_stride_, hand_stride_);
  }

  [[nodiscard]] size_t board_count() const { return board_count_; }
  [[nodiscard]] size_t hand_stride() const { return hand_stride_; }

private:
  std::span<T> storage_;
  size_t board_count_;
  size_t hand_stride_;
};

template <typename T>
class SlotBoardHandView {
public:
  SlotBoardHandView(std::span<T> storage, size_t slot_count, size_t board_count,
                    size_t hand_stride)
      : storage_(storage), slot_count_(slot_count), board_count_(board_count),
        hand_stride_(hand_stride) {
    assert(storage_.size() >= slot_count_ * board_count_ * hand_stride_);
  }

  [[nodiscard]] T &operator()(size_t slot, size_t board, size_t hand) const {
    assert(slot < slot_count_);
    assert(board < board_count_);
    assert(hand < hand_stride_);
    return storage_[((slot * board_count_) + board) * hand_stride_ + hand];
  }

  [[nodiscard]] std::span<T> hands(size_t slot, size_t board) const {
    assert(slot < slot_count_);
    assert(board < board_count_);
    const size_t offset = ((slot * board_count_) + board) * hand_stride_;
    return storage_.subspan(offset, hand_stride_);
  }

  [[nodiscard]] size_t slot_count() const { return slot_count_; }
  [[nodiscard]] size_t board_count() const { return board_count_; }
  [[nodiscard]] size_t hand_stride() const { return hand_stride_; }

private:
  std::span<T> storage_;
  size_t slot_count_;
  size_t board_count_;
  size_t hand_stride_;
};

struct StreetBatchView {
  std::span<const BoardIndex> boards;
  std::array<std::span<const float>, 2> root_reaches;
  std::span<float> root_values;
  std::span<const float> cumulative_chance_reaches;

  [[nodiscard]] size_t board_count() const { return boards.size(); }
};

struct OwnedStreetBatch {
  std::vector<BoardIndex> boards;
  std::array<std::vector<float>, 2> root_reaches;
  std::vector<float> root_values;
  std::vector<float> cumulative_chance_reaches;

  [[nodiscard]] StreetBatchView view();
};

struct RiverTerminalOperator {
  virtual ~RiverTerminalOperator() = default;

  virtual std::span<const Hand> hands(Player player) const = 0;

  virtual void mask_board_reaches(BoardIndex board, Player player,
                                  std::span<const float> input,
                                  std::span<float> output) const;
  virtual void mask_board_values(BoardIndex board, Player player,
                                 std::span<float> values) const;

  virtual void evaluate_folds(BoardIndex board, Player evaluated_player,
                              std::span<const float> opponent_reaches,
                              size_t opponent_hand_stride,
                              std::span<const float> payoffs,
                              std::span<float> values,
                              size_t evaluated_hand_stride) const = 0;

  virtual void evaluate_showdowns(BoardIndex board, Player evaluated_player,
                                  std::span<const float> opponent_reaches,
                                  size_t opponent_hand_stride,
                                  std::span<const CompiledShowdown> showdowns,
                                  std::span<float> values,
                                  size_t evaluated_hand_stride) const = 0;

  virtual void reserve_showdown_summary_scratch(
      std::span<const BoardIndex> boards, size_t showdown_count) const {}

  // Batch layouts are [terminal endpoint][board][hand].
  virtual void evaluate_fold_board_batch(
      std::span<const BoardIndex> boards, Player evaluated_player,
      std::span<const float> opponent_reaches, size_t opponent_hand_stride,
      std::span<const float> payoffs, std::span<float> values,
      size_t evaluated_hand_stride) const;

  virtual void evaluate_showdown_board_batch(
      std::span<const BoardIndex> boards, Player evaluated_player,
      std::span<const float> opponent_reaches, size_t opponent_hand_stride,
      std::span<const CompiledShowdown> showdowns, std::span<float> values,
      size_t evaluated_hand_stride) const;
};

void propagate_reaches(
    const StreetTree &game, BoardIndex board,
    const std::array<std::span<const float>, 2> &root_reaches,
    StreetReachBuffers &outputs);

void validate_batch(const StreetBatchView &batch, const CompiledStreet &compiled,
                    Player evaluated_player);

void propagate_reaches_batch(
    const StreetTree &game, const StreetBatchView &batch,
    std::array<std::span<float>, 2> reach_workspaces);

void update_player(StreetTree &game, BoardIndex board, Player player,
                   std::span<const float> reach_workspace,
                   StreetValueBuffer &values, std::span<float> root_values,
                   float chance_reach = 1.0F, float iteration_weight = 1.0F);

[[nodiscard]] std::span<const float>
fold_reaches(const StreetTree &game, const StreetReachBuffers &buffers,
             Player player);
[[nodiscard]] std::span<const float>
showdown_reaches(const StreetTree &game, const StreetReachBuffers &buffers,
                 Player player);
[[nodiscard]] std::span<const float>
boundary_reaches(const StreetTree &game, const StreetReachBuffers &buffers,
                 Player player);

void prepare_value_buffer(const StreetTree &game, Player player,
                          StreetValueBuffer &values);
[[nodiscard]] std::span<float>
fold_values(const StreetTree &game, StreetValueBuffer &values, Player player);
[[nodiscard]] std::span<float> showdown_values(const StreetTree &game,
                                               StreetValueBuffer &values,
                                               Player player);
[[nodiscard]] std::span<float> boundary_values(const StreetTree &game,
                                               StreetValueBuffer &values,
                                               Player player);

void update_river(StreetTree &river, BoardIndex board,
                  const std::array<std::span<const float>, 2> &root_reaches,
                  Player player, const RiverTerminalOperator &terminals,
                  RiverUpdateBuffers &buffers, std::span<float> root_values,
                  float chance_reach = 1.0F, float iteration_weight = 1.0F);
} // namespace spmv_poker
