#pragma once

#include "spmv_poker/spmv_tree.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <span>
#include <vector>

namespace spmv_poker {

constexpr uint32_t INVALID_COMPILED_INDEX = UINT32_MAX;

/*
 * StreetTopology is convenient for building and inspecting a betting tree, but
 * evaluating it requires repeatedly figuring out where each node and action
 * reads from and writes to. The compiler does that work once and produces a
 * simpler representation that the evaluator can follow directly
 */
struct CompiledForwardAction {
  uint32_t destination_slot;
  uint32_t cfr_action_offset;
};

struct CompiledBackwardAction {
  uint32_t source_slot;
  uint32_t cfr_action_offset;
};

struct CompiledForwardNode {
  uint32_t input_reach_slot;
  uint32_t cfr_offset_within_board;
  uint32_t action_begin;
  uint16_t action_count;
  Player player;
};

struct CompiledBackwardNode {
  uint32_t output_value_slot;
  uint32_t cfr_offset_within_board;
  uint32_t retained_reach_slot;
  uint32_t action_begin;
  uint16_t action_count;
  Player player;
};

struct CompiledShowdown {
  float win_payoff;
  float loss_payoff;
};

struct CompiledForwardPlan {
  // Each player gets a separate forward plan because reaches at their own
  // decision nodes must be retained, while reaches at opponent nodes are
  // temporary
  std::vector<CompiledForwardNode> nodes;
  std::vector<CompiledForwardAction> actions;

  uint32_t root_input_reach_slot;

  uint32_t temporary_reach_count;
  uint32_t retained_reach_begin;
  uint32_t retained_reach_count;
  uint32_t fold_reach_begin;
  uint32_t fold_reach_count;
  uint32_t showdown_reach_begin;
  uint32_t showdown_reach_count;
  uint32_t boundary_reach_begin;
  uint32_t boundary_reach_count;
  uint32_t workspace_slot_count;
};

struct CompiledStreet {
  // The backward plan is shared because action-value sources do not change
  // based on which player is being updated
  std::vector<CompiledBackwardNode> nodes;
  std::vector<CompiledBackwardAction> backward_actions;

  std::array<CompiledForwardPlan, 2> forward_plans;

  uint32_t root_value_slot;
  uint32_t fold_value_begin;
  uint32_t fold_value_count;
  uint32_t showdown_value_begin;
  uint32_t showdown_value_count;
  uint32_t boundary_value_begin;
  uint32_t boundary_value_count;
  uint32_t value_workspace_slot_count;
  std::array<uint32_t, 2> retained_reach_counts;
  Street street;
  std::array<uint32_t, 2> hand_counts;
  std::array<uint32_t, 2> padded_hand_counts;
  uint32_t state_entries_per_board;
  std::vector<float> fold_payoffs;
  std::vector<CompiledShowdown> showdowns;
};

struct CompiledTransitionGraph {
  Street parent_street;
  Street child_street;
  std::vector<uint32_t> child_offsets;
  std::vector<BoardIndex> child_boards;
  std::vector<uint8_t> dealt_cards;
  std::vector<float> local_chance_weights;
};

struct TurnGroup {
  uint32_t parent_board_begin = 0;
  uint32_t parent_board_count = 0;
  uint32_t child_board_count = 0;
};

struct RiverGroup {
  uint32_t local_parent_board_begin = 0;
  uint32_t parent_board_count = 0;
  uint32_t child_board_count = 0;
};

struct ExecutionSchedule {
  size_t turn_board_capacity = 0;
  size_t river_board_capacity = 0;
  std::vector<TurnGroup> turn_groups;
  std::vector<std::vector<RiverGroup>> river_groups_by_turn_group;

  [[nodiscard]] std::span<const RiverGroup>
  river_groups_for(size_t turn_group_index) const {
    return river_groups_by_turn_group[turn_group_index];
  }
};

static_assert(std::is_trivially_copyable_v<CompiledForwardAction>);
static_assert(std::is_trivially_copyable_v<CompiledBackwardAction>);
static_assert(std::is_trivially_copyable_v<CompiledForwardNode>);
static_assert(std::is_trivially_copyable_v<CompiledBackwardNode>);
static_assert(std::is_trivially_copyable_v<CompiledShowdown>);

[[nodiscard]] CompiledStreet compile_street(const StreetTopology &topology);
[[nodiscard]] CompiledTransitionGraph
compile_transition_graph(const RunoutIndex &board_index, Street parent_street);
[[nodiscard]] ExecutionSchedule
build_execution_schedule(const RunoutIndex &board_index, Street starting_street,
                         size_t turn_board_capacity,
                         size_t river_board_capacity);

struct StreetTree {
  const CompiledStreet compiled;
  const RunoutIndex board_index;
  std::vector<float> regrets;
  std::vector<float> cumulative_strategy;

  StreetTree(StreetTopology topology, uint64_t flop_mask);
  StreetTree(StreetTopology topology, uint64_t public_mask,
             size_t public_card_count);
  StreetTree(CompiledStreet compiled, RunoutIndex board_index,
             std::vector<float> regrets,
             std::vector<float> cumulative_strategy);

  [[nodiscard]] size_t state_entry(BoardIndex board, uint32_t node_slot,
                                   size_t hand, size_t action) const;
  [[nodiscard]] std::span<float> board_regrets(BoardIndex board);
  [[nodiscard]] std::span<const float> board_regrets(BoardIndex board) const;
  [[nodiscard]] std::span<float> board_cumulative_strategy(BoardIndex board);
  [[nodiscard]] std::span<const float>
  board_cumulative_strategy(BoardIndex board) const;
};

} // namespace spmv_poker
