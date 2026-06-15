#pragma once

#include "spmv_poker/tree_compiler.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdint>
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

struct BatchJob {
  BoardIndex board;
  std::array<std::span<const float>, 2> root_reaches;
  std::span<float> root_values;
  float chance_reach = 1.0F;
};

struct BatchBuffers {
  // Dense [workspace slot][board][hand] layouts.
  std::array<std::vector<float>, 2> reaches;
  std::vector<float> values;
};

struct BatchContext;

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

void update_batch(StreetTree &river, std::span<const BatchJob> jobs,
                  Player player, const RiverTerminalOperator &terminals,
                  BatchContext &context, float iteration_weight = 1.0F);

struct BatchFrame {
  BatchBuffers buffers;
  std::vector<BoardIndex> boards;
  std::vector<std::vector<BoardIndex>> child_boards_cache;
  std::vector<BatchJob> child_jobs;
  std::vector<float> child_root_values;
  std::vector<float> player_root_masks;
  std::vector<float> opponent_root_masks;
};

struct BatchContext {
  std::array<BatchFrame, 3> frames;
};

template <Street CurrentStreet, Street StartingStreet>
void update_street_chain(
    StreetTree &street, std::span<StreetTree> future_street_trees,
    std::span<const BatchJob> jobs, Player player,
    const RiverTerminalOperator &terminals, BatchContext &context,
    float iteration_weight = 1.0F, size_t boundary_batch_size = 0);

void update_turn(StreetTree &turn, std::span<StreetTree> future_street_trees,
                 std::span<const BatchJob> jobs, Player player,
                 const RiverTerminalOperator &terminals, BatchContext &context,
                 float iteration_weight = 1.0F,
                 size_t boundary_batch_size = 0);

// Solves a chance root by evaluating each child board and folding the child
// root values through the board probabilities
void update_chance_root(StreetTree &river, std::span<const BatchJob> jobs,
                        Player player, const RiverTerminalOperator &terminals,
                        BatchContext &context, std::span<float> root_values,
                        float iteration_weight = 1.0F);

template <Street CurrentStreet, Street StartingStreet>
void update_street_chain(
    StreetTree &street, std::span<StreetTree> future_street_trees,
    std::span<const BatchJob> jobs, Player player,
    const RiverTerminalOperator &terminals, BatchContext &context,
    float iteration_weight, size_t boundary_batch_size) {
  static_assert(static_cast<int>(CurrentStreet) >=
                static_cast<int>(StartingStreet));
  constexpr size_t frame_index =
      static_cast<size_t>(CurrentStreet) - static_cast<size_t>(StartingStreet);
  auto &frame = context.frames[frame_index];
  if (jobs.empty()) {
    return;
  }

  const CompiledStreet &compiled = street.compiled;
  size_t board_count = jobs.size();
  size_t player_index = static_cast<size_t>(player);
  size_t opponent_index = static_cast<size_t>(opponent(player));
  size_t player_width = compiled.padded_hand_counts[player_index];
  size_t opponent_width = compiled.padded_hand_counts[opponent_index];
  frame.boards.reserve(board_count);
  frame.boards.resize(board_count);
  for (size_t board = 0; board < board_count; ++board) {
    frame.boards[board] = jobs[board].board;
  }
  frame.child_boards_cache.resize(
      street.board_index.board_count(compiled.street));

  if constexpr (CurrentStreet == Street::River) {
    assert(future_street_trees.empty());
  }
  for (Player side : {Player::Hero, Player::Villain}) {
    size_t propagated_index = static_cast<size_t>(side);
    size_t width = compiled.padded_hand_counts[propagated_index];
    const CompiledForwardPlan &plan =
        compiled.forward_plans[propagated_index];
    std::vector<float> &workspace = frame.buffers.reaches[propagated_index];
    workspace.reserve(static_cast<size_t>(plan.workspace_slot_count) *
                      board_count * width);
    workspace.assign(static_cast<size_t>(plan.workspace_slot_count) *
                         board_count * width,
                     0.0F);

    for (size_t board = 0; board < board_count; ++board) {
      assert(jobs[board].root_reaches[propagated_index].size() == width);
      StreetReachBuffers board_reaches;
      std::array<std::span<const float>, 2> root_reaches{};
      root_reaches[0] = jobs[board].root_reaches[0];
      root_reaches[1] = jobs[board].root_reaches[1];
      propagate_reaches(street, jobs[board].board, root_reaches,
                        board_reaches);
      const std::vector<float> &board_workspace =
          board_reaches.workspaces[propagated_index];
      for (size_t slot = 0; slot < plan.workspace_slot_count; ++slot) {
        const float *source =
            board_workspace.data() + static_cast<size_t>(slot) * width;
        float *destination =
            workspace.data() +
            (static_cast<size_t>(slot) * board_count + board) * width;
        std::copy(source, source + width, destination);
      }
    }
  }
  frame.buffers.values.reserve(static_cast<size_t>(compiled.value_workspace_slot_count) *
                              board_count * player_width);
  frame.buffers.values.assign(static_cast<size_t>(compiled.value_workspace_slot_count) *
                                  board_count * player_width,
                              0.0F);
  const CompiledForwardPlan &opponent_plan =
      compiled.forward_plans[opponent_index];
  const std::vector<float> &opponent_reaches =
      frame.buffers.reaches[opponent_index];

  terminals.reserve_showdown_summary_scratch(
      frame.boards, static_cast<size_t>(compiled.showdowns.size()));
  terminals.evaluate_fold_board_batch(
      frame.boards, player,
      std::span<const float>(opponent_reaches)
          .subspan(static_cast<size_t>(opponent_plan.fold_reach_begin) *
                       board_count * opponent_width,
                   static_cast<size_t>(opponent_plan.fold_reach_count) *
                       board_count * opponent_width),
      opponent_width, compiled.fold_payoffs,
      std::span<float>(frame.buffers.values)
          .subspan(static_cast<size_t>(compiled.fold_value_begin) *
                       board_count * player_width,
                   static_cast<size_t>(compiled.fold_value_count) *
                       board_count * player_width),
      player_width);
  if constexpr (CurrentStreet == Street::River) {
    terminals.evaluate_showdown_board_batch(
        frame.boards, player,
        std::span<const float>(opponent_reaches)
            .subspan(static_cast<size_t>(opponent_plan.showdown_reach_begin) *
                         board_count * opponent_width,
                     static_cast<size_t>(opponent_plan.showdown_reach_count) *
                         board_count * opponent_width),
        opponent_width, compiled.showdowns,
        std::span<float>(frame.buffers.values)
            .subspan(static_cast<size_t>(compiled.showdown_value_begin) *
                         board_count * player_width,
                     static_cast<size_t>(compiled.showdown_value_count) *
                         board_count * player_width),
        player_width);
  }

  if constexpr (CurrentStreet != Street::River) {
    assert(!future_street_trees.empty());
    constexpr Street next = detail::next_street(CurrentStreet);
    StreetTree &next_tree = future_street_trees.front();
    assert(next_tree.compiled.street == next);
    std::span<StreetTree> remaining = future_street_trees.subspan(1);

    if (compiled.boundary_value_count > 0) {
      size_t child_board_count = street.board_index.child_count(compiled.street);
      frame.child_jobs.reserve(child_board_count);
      frame.child_root_values.reserve(child_board_count * player_width);
      frame.player_root_masks.reserve(child_board_count * player_width);
      frame.opponent_root_masks.reserve(child_board_count * opponent_width);
      frame.child_jobs.resize(child_board_count);
      frame.child_root_values.resize(child_board_count * player_width);
      frame.player_root_masks.resize(child_board_count * player_width);
      frame.opponent_root_masks.resize(child_board_count * opponent_width);

      StreetReachBuffers boundary_reach_buffers{frame.buffers.reaches};
      std::span<const float> player_boundary_reaches =
          boundary_reaches(street, boundary_reach_buffers, player);
      std::span<const float> opponent_boundary_reaches =
          boundary_reaches(street, boundary_reach_buffers, opponent(player));

      for (size_t board = 0; board < board_count; ++board) {
        uint64_t board_mask =
            street.board_index.board_mask(compiled.street, jobs[board].board);
        std::vector<BoardIndex> &child_boards =
            frame.child_boards_cache[jobs[board].board];
        if (child_boards.empty()) {
          child_boards.reserve(child_board_count);
          for (uint8_t dealt_card = 0; dealt_card < DECK_SIZE; ++dealt_card) {
            if ((board_mask & card_mask(dealt_card)) != 0) {
              continue;
            }
            child_boards.push_back(street.board_index.child_board(
                compiled.street, jobs[board].board, dealt_card));
          }
        }
        assert(child_boards.size() == child_board_count);

        for (size_t endpoint = 0; endpoint < compiled.boundary_value_count;
             ++endpoint) {
          size_t boundary_index = endpoint * board_count + board;
          std::span<const float> player_boundary =
              player_boundary_reaches.subspan(boundary_index * player_width,
                                              player_width);
          std::span<const float> opponent_boundary =
              opponent_boundary_reaches.subspan(
                  boundary_index *
                      compiled.padded_hand_counts[opponent_index],
                  compiled.padded_hand_counts[opponent_index]);

          for (size_t child_board_offset = 0;
               child_board_offset < child_boards.size();
               ++child_board_offset) {
            std::array<std::span<const float>, 2> root_reaches{};
            std::span<float> masked_player_root =
                std::span<float>(frame.player_root_masks)
                    .subspan(child_board_offset * player_width, player_width);
            std::span<float> masked_opponent_root =
                std::span<float>(frame.opponent_root_masks)
                    .subspan(child_board_offset * opponent_width,
                             opponent_width);
            terminals.mask_board_reaches(
                child_boards[child_board_offset], player, player_boundary,
                masked_player_root);
            terminals.mask_board_reaches(
                child_boards[child_board_offset], opponent(player),
                opponent_boundary, masked_opponent_root);
            root_reaches[player_index] = masked_player_root;
            root_reaches[opponent_index] = masked_opponent_root;
            frame.child_jobs[child_board_offset] = {
                .board = child_boards[child_board_offset],
                .root_reaches = root_reaches,
                .root_values =
                    std::span<float>(frame.child_root_values)
                        .subspan(child_board_offset * player_width,
                                 player_width),
                .chance_reach =
                    1.0F / static_cast<float>(child_board_count),
            };
          }

          size_t effective_batch_size =
              boundary_batch_size == 0 ? frame.child_jobs.size()
                                       : boundary_batch_size;
          for (size_t begin = 0; begin < frame.child_jobs.size();
               begin += effective_batch_size) {
            size_t count = std::min(effective_batch_size,
                                    frame.child_jobs.size() - begin);
            update_street_chain<next, StartingStreet>(
                next_tree, remaining,
                std::span<const BatchJob>(frame.child_jobs)
                    .subspan(begin, count),
                player, terminals, context, iteration_weight,
                boundary_batch_size);
          }
          float *parent_values =
              frame.buffers.values.data() +
              static_cast<size_t>(compiled.boundary_value_begin) * board_count *
                  player_width +
              boundary_index * player_width;
          for (const BatchJob &child_job : frame.child_jobs) {
            for (size_t hand = 0; hand < player_width; ++hand) {
              parent_values[hand] +=
                  child_job.chance_reach * child_job.root_values[hand];
            }
          }
        }
      }
    }
  }
  const CompiledForwardPlan &acting_plan =
      compiled.forward_plans[player_index];
  for (const CompiledBackwardNode &node : compiled.nodes) {
    for (size_t board = 0; board < board_count; ++board) {
      std::span<float> board_regrets = street.board_regrets(jobs[board].board);
      std::span<float> board_strategy =
          street.board_cumulative_strategy(jobs[board].board);
      for (size_t hand_begin = 0; hand_begin < player_width;
           hand_begin += HAND_BLOCK_SIZE) {
        detail::HandBlock output = detail::batched_hand_block_at_slot(
            std::span<float>(frame.buffers.values), node.output_value_slot,
            board, board_count, player_width, hand_begin);
        std::ranges::fill(output, 0.0F);
        bool updates_player = node.player == player;
        alignas(64) std::array<float, HAND_BLOCK_SIZE> inverse_positive_sums;
        if (updates_player) {
          detail::compute_regret_matching_factors(
              std::span<const float>(board_regrets),
              node.cfr_offset_within_board, node.action_begin,
              node.action_count, hand_begin,
              std::span<const CompiledBackwardAction>(
                  compiled.backward_actions),
              inverse_positive_sums);
        }
        float uniform_probability =
            1.0F / static_cast<float>(node.action_count);
        for (size_t action = 0; action < node.action_count; ++action) {
          const CompiledBackwardAction &compiled_action =
              compiled.backward_actions[node.action_begin + action];
          detail::ConstHandBlock source = detail::batched_hand_block_at_slot(
              std::span<const float>(frame.buffers.values),
              compiled_action.source_slot, board, board_count, player_width,
              hand_begin);
          if (updates_player) {
            const float *regrets =
                board_regrets.data() + node.cfr_offset_within_board +
                compiled_action.cfr_action_offset + hand_begin;
            float *strategy_sum =
                board_strategy.data() + node.cfr_offset_within_board +
                compiled_action.cfr_action_offset + hand_begin;
            detail::ConstHandBlock player_reach =
                detail::batched_hand_block_at_slot(
                std::span<const float>(frame.buffers.reaches[player_index]),
                    acting_plan.retained_reach_begin + node.retained_reach_slot,
                    board, board_count, player_width, hand_begin);
            for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
              float inverse = inverse_positive_sums[lane];
              float strategy =
                  inverse > 0.0F ? std::max(0.0F, regrets[lane]) * inverse
                                 : uniform_probability;
              output[lane] += strategy * source[lane];
              strategy_sum[lane] +=
                  iteration_weight * player_reach[lane] * strategy;
            }
          } else {
            for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
              output[lane] += source[lane];
            }
          }
        }
        if (updates_player) {
          for (size_t action = 0; action < node.action_count; ++action) {
            const CompiledBackwardAction &compiled_action =
                compiled.backward_actions[node.action_begin + action];
            detail::ConstHandBlock source = detail::batched_hand_block_at_slot(
                std::span<const float>(frame.buffers.values),
                compiled_action.source_slot, board, board_count, player_width,
                hand_begin);
            float *regrets =
                board_regrets.data() + node.cfr_offset_within_board +
                compiled_action.cfr_action_offset + hand_begin;
            for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
              regrets[lane] =
                  std::max(0.0F, regrets[lane] +
                                     jobs[board].chance_reach *
                                         (source[lane] - output[lane]));
            }
          }
        }
      }
    }
  }

  for (size_t board = 0; board < board_count; ++board) {
    assert(jobs[board].root_values.size() == player_width);
    for (size_t hand_begin = 0; hand_begin < player_width;
         hand_begin += HAND_BLOCK_SIZE) {
      detail::ConstHandBlock root = detail::batched_hand_block_at_slot(
          std::span<const float>(frame.buffers.values), compiled.root_value_slot,
          board, board_count, player_width, hand_begin);
      std::ranges::copy(root, jobs[board].root_values.begin() + hand_begin);
    }
    if constexpr (CurrentStreet == Street::River) {
      terminals.mask_board_values(jobs[board].board, player,
                                  jobs[board].root_values);
    } else {
      uint64_t board_mask =
          street.board_index.board_mask(compiled.street, jobs[board].board);
      std::span<const Hand> hands = terminals.hands(player);
      for (size_t hand = 0; hand < hands.size(); ++hand) {
        if ((hands[hand].mask & board_mask) != 0) {
          jobs[board].root_values[hand] = 0.0F;
        }
      }
    }
  }
}
} // namespace spmv_poker
