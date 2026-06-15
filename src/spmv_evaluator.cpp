#include "spmv_poker/spmv_evaluator.h"
#include "spmv_poker/spmv_terminal.h"

#include <algorithm>
#include <array>
#include <cassert>

namespace spmv_poker {

namespace {

using HandBlock = std::span<float, HAND_BLOCK_SIZE>;
using ConstHandBlock = std::span<const float, HAND_BLOCK_SIZE>;

HandBlock hand_block_at_slot(std::span<float> storage, uint32_t slot,
                             size_t hands_per_slot, size_t hand_begin) {
  return HandBlock(storage.data() + static_cast<size_t>(slot) * hands_per_slot +
                       hand_begin,
                   HAND_BLOCK_SIZE);
}

ConstHandBlock hand_block_at_slot(std::span<const float> storage, uint32_t slot,
                                  size_t hands_per_slot, size_t hand_begin) {
  return ConstHandBlock(
      storage.data() + static_cast<size_t>(slot) * hands_per_slot + hand_begin,
      HAND_BLOCK_SIZE);
}

HandBlock batched_hand_block_at_slot(std::span<float> storage, uint32_t slot,
                                     size_t board, size_t board_count,
                                     size_t hands_per_slot,
                                     size_t hand_begin) {
  return HandBlock(
      storage.data() +
          (static_cast<size_t>(slot) * board_count + board) * hands_per_slot +
          hand_begin,
      HAND_BLOCK_SIZE);
}

ConstHandBlock batched_hand_block_at_slot(std::span<const float> storage,
                                          uint32_t slot, size_t board,
                                          size_t board_count,
                                          size_t hands_per_slot,
                                          size_t hand_begin) {
  return ConstHandBlock(
      storage.data() +
          (static_cast<size_t>(slot) * board_count + board) * hands_per_slot +
          hand_begin,
      HAND_BLOCK_SIZE);
}

template <typename Action>
void compute_regret_matching_factors(std::span<const float> board_regrets,
                                     uint32_t cfr_offset_within_board,
                                     uint32_t action_begin,
                                     uint16_t action_count, size_t hand_begin,
                                     std::span<const Action> actions,
                                     HandBlock inverse_positive_sums) {
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

} // namespace

void propagate_block(const StreetTree &game, BoardIndex board, Player player,
                     size_t hand_begin, ConstHandBlock root,
                     std::span<float> workspace) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  size_t width = compiled.padded_hand_counts[player_index];
  const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
  std::span<const float> board_regrets = game.board_regrets(board);

  std::copy(root.begin(), root.end(),
            hand_block_at_slot(workspace, plan.root_input_reach_slot, width,
                               hand_begin)
                .begin());

  for (size_t node_offset = plan.nodes.size(); node_offset-- > 0;) {
    const CompiledForwardNode &node = plan.nodes[node_offset];
    ConstHandBlock input =
        hand_block_at_slot(std::span<const float>(workspace),
                           node.input_reach_slot, width, hand_begin);
    alignas(64) std::array<float, HAND_BLOCK_SIZE> inverse_positive_sums;
    if (node.player == player) {
      compute_regret_matching_factors(
          board_regrets, node.cfr_offset_within_board, node.action_begin,
          node.action_count, hand_begin,
          std::span<const CompiledForwardAction>(plan.actions),
          inverse_positive_sums);
    }

    float uniform_probability = 1.0F / static_cast<float>(node.action_count);
    for (size_t action = 0; action < node.action_count; ++action) {
      const CompiledForwardAction &compiled_action =
          plan.actions[node.action_begin + action];

      HandBlock destination = hand_block_at_slot(
          workspace, compiled_action.destination_slot, width, hand_begin);

      if (node.player == player) {
        const float *regrets = board_regrets.data() +
                               node.cfr_offset_within_board +
                               compiled_action.cfr_action_offset + hand_begin;
        for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
          float inverse_positive_sum = inverse_positive_sums[lane];
          float strategy =
              inverse_positive_sum > 0.0F
                  ? std::max(0.0F, regrets[lane]) * inverse_positive_sum
                  : uniform_probability;
          destination[lane] += input[lane] * strategy;
        }
      } else {
        for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
          destination[lane] += input[lane];
        }
      }
    }
  }
}

void propagate_reaches(
    const StreetTree &game, BoardIndex board,
    const std::array<std::span<const float>, 2> &root_reaches,
    StreetReachBuffers &outputs) {
  const CompiledStreet &compiled = game.compiled;
  for (Player player : {Player::Hero, Player::Villain}) {
    size_t player_index = static_cast<size_t>(player);
    size_t width = compiled.padded_hand_counts[player_index];
    const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
    assert(root_reaches[player_index].size() == width);
    outputs.workspaces[player_index].assign(
        static_cast<size_t>(plan.workspace_slot_count) * width, 0.0F);

    for (size_t hand_begin = 0; hand_begin < width;
         hand_begin += HAND_BLOCK_SIZE) {
      ConstHandBlock root(root_reaches[player_index].data() + hand_begin,
                          HAND_BLOCK_SIZE);
      propagate_block(game, board, player, hand_begin, root,
                      outputs.workspaces[player_index]);
    }
  }
}

void update_player_block(StreetTree &game, BoardIndex board, Player player,
                         size_t hand_begin,
                         std::span<const float> reach_workspace,
                         std::span<float> value_workspace,
                         HandBlock root_values, float chance_reach,
                         float iteration_weight) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  size_t width = compiled.padded_hand_counts[player_index];
  const CompiledForwardPlan &forward_plan =
      compiled.forward_plans[player_index];
  std::span<float> board_regrets = game.board_regrets(board);
  std::span<float> board_cumulative_strategy =
      game.board_cumulative_strategy(board);

  for (const CompiledBackwardNode &node : compiled.nodes) {
    HandBlock output = hand_block_at_slot(
        value_workspace, node.output_value_slot, width, hand_begin);
    std::ranges::fill(output, 0.0F);
    bool updates_player = node.player == player;
    alignas(64) std::array<float, HAND_BLOCK_SIZE> inverse_positive_sums;
    if (updates_player) {
      compute_regret_matching_factors(
          std::span<const float>(board_regrets), node.cfr_offset_within_board,
          node.action_begin, node.action_count, hand_begin,
          std::span<const CompiledBackwardAction>(compiled.backward_actions),
          inverse_positive_sums);
    }

    float uniform_probability = 1.0F / static_cast<float>(node.action_count);
    for (size_t action = 0; action < node.action_count; ++action) {
      const CompiledBackwardAction &compiled_action =
          compiled.backward_actions[node.action_begin + action];

      ConstHandBlock source =
          hand_block_at_slot(std::span<const float>(value_workspace),
                             compiled_action.source_slot, width, hand_begin);
      if (updates_player) {
        const float *regrets = board_regrets.data() +
                               node.cfr_offset_within_board +
                               compiled_action.cfr_action_offset + hand_begin;
        float *cumulative_strategy =
            board_cumulative_strategy.data() + node.cfr_offset_within_board +
            compiled_action.cfr_action_offset + hand_begin;

        ConstHandBlock player_reach = hand_block_at_slot(
            reach_workspace,
            forward_plan.retained_reach_begin + node.retained_reach_slot, width,
            hand_begin);

        for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
          float inverse_positive_sum = inverse_positive_sums[lane];
          float strategy =
              inverse_positive_sum > 0.0F
                  ? std::max(0.0F, regrets[lane]) * inverse_positive_sum
                  : uniform_probability;
          output[lane] += strategy * source[lane];
          cumulative_strategy[lane] +=
              iteration_weight * player_reach[lane] * strategy;
        }
      } else {
        for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
          output[lane] += source[lane];
        }
      }
    }

    if (!updates_player) {
      continue;
    }
    for (size_t action = 0; action < node.action_count; ++action) {
      const CompiledBackwardAction &compiled_action =
          compiled.backward_actions[node.action_begin + action];
      ConstHandBlock source =
          hand_block_at_slot(std::span<const float>(value_workspace),
                             compiled_action.source_slot, width, hand_begin);
      float *regrets = board_regrets.data() + node.cfr_offset_within_board +
                       compiled_action.cfr_action_offset + hand_begin;
      for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
        regrets[lane] = std::max(
            0.0F, regrets[lane] + chance_reach * (source[lane] - output[lane]));
      }
    }
  }

  ConstHandBlock root_value =
      hand_block_at_slot(std::span<const float>(value_workspace),
                         compiled.root_value_slot, width, hand_begin);
  std::copy(root_value.begin(), root_value.end(), root_values.begin());
}

void update_player(StreetTree &game, BoardIndex board, Player player,
                   std::span<const float> reach_workspace,
                   StreetValueBuffer &values, std::span<float> root_values,
                   float chance_reach, float iteration_weight) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  size_t width = compiled.padded_hand_counts[player_index];
  const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
  assert(reach_workspace.size() ==
         static_cast<size_t>(plan.workspace_slot_count) * width);
  assert(values.workspace.size() ==
         static_cast<size_t>(compiled.value_workspace_slot_count) * width);
  assert(root_values.size() == width);

  std::fill_n(values.workspace.begin(),
              static_cast<size_t>(compiled.nodes.size()) * width, 0.0F);
  for (size_t hand_begin = 0; hand_begin < width;
       hand_begin += HAND_BLOCK_SIZE) {
    HandBlock root_value_block(root_values.data() + hand_begin,
                               HAND_BLOCK_SIZE);
    update_player_block(game, board, player, hand_begin, reach_workspace,
                        values.workspace, root_value_block, chance_reach,
                        iteration_weight);
  }
}

std::span<const float> fold_reaches(const StreetTree &game,
                                    const StreetReachBuffers &buffers,
                                    Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
  size_t width = compiled.padded_hand_counts[player_index];
  return std::span<const float>(buffers.workspaces[player_index])
      .subspan(static_cast<size_t>(plan.fold_reach_begin) * width,
               static_cast<size_t>(plan.fold_reach_count) * width);
}

std::span<const float> showdown_reaches(const StreetTree &game,
                                        const StreetReachBuffers &buffers,
                                        Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
  size_t width = compiled.padded_hand_counts[player_index];
  return std::span<const float>(buffers.workspaces[player_index])
      .subspan(static_cast<size_t>(plan.showdown_reach_begin) * width,
               static_cast<size_t>(plan.showdown_reach_count) * width);
}

std::span<const float> boundary_reaches(const StreetTree &game,
                                        const StreetReachBuffers &buffers,
                                        Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t player_index = static_cast<size_t>(player);
  const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
  size_t width = compiled.padded_hand_counts[player_index];
  return std::span<const float>(buffers.workspaces[player_index])
      .subspan(static_cast<size_t>(plan.boundary_reach_begin) * width,
               static_cast<size_t>(plan.boundary_reach_count) * width);
}

void prepare_value_buffer(const StreetTree &game, Player player,
                          StreetValueBuffer &values) {
  const CompiledStreet &compiled = game.compiled;
  size_t width = compiled.padded_hand_counts[static_cast<size_t>(player)];
  values.workspace.assign(
      static_cast<size_t>(compiled.value_workspace_slot_count) * width, 0.0F);
}

std::span<float> fold_values(const StreetTree &game, StreetValueBuffer &values,
                             Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t width = compiled.padded_hand_counts[static_cast<size_t>(player)];
  return std::span<float>(values.workspace)
      .subspan(static_cast<size_t>(compiled.fold_value_begin) * width,
               static_cast<size_t>(compiled.fold_value_count) * width);
}

std::span<float> showdown_values(const StreetTree &game,
                                 StreetValueBuffer &values, Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t width = compiled.padded_hand_counts[static_cast<size_t>(player)];
  return std::span<float>(values.workspace)
      .subspan(static_cast<size_t>(compiled.showdown_value_begin) * width,
               static_cast<size_t>(compiled.showdown_value_count) * width);
}

std::span<float> boundary_values(const StreetTree &game,
                                 StreetValueBuffer &values, Player player) {
  const CompiledStreet &compiled = game.compiled;
  size_t width = compiled.padded_hand_counts[static_cast<size_t>(player)];
  return std::span<float>(values.workspace)
      .subspan(static_cast<size_t>(compiled.boundary_value_begin) * width,
               static_cast<size_t>(compiled.boundary_value_count) * width);
}

void update_river(StreetTree &river, BoardIndex board,
                  const std::array<std::span<const float>, 2> &root_reaches,
                  Player player, const RiverTerminalOperator &terminals,
                  RiverUpdateBuffers &buffers, std::span<float> root_values,
                  float chance_reach, float iteration_weight) {
  const CompiledStreet &compiled = river.compiled;
  assert(compiled.street == Street::River);
  assert(compiled.boundary_value_count == 0);

  propagate_reaches(river, board, root_reaches, buffers.reaches);
  prepare_value_buffer(river, player, buffers.values);

  size_t player_index = static_cast<size_t>(player);
  size_t opponent_index = static_cast<size_t>(opponent(player));
  size_t player_width = compiled.padded_hand_counts[player_index];
  size_t opponent_width = compiled.padded_hand_counts[opponent_index];
  terminals.evaluate_folds(
      board, player, fold_reaches(river, buffers.reaches, opponent(player)),
      opponent_width, compiled.fold_payoffs,
      fold_values(river, buffers.values, player), player_width);
  terminals.evaluate_showdowns(
      board, player, showdown_reaches(river, buffers.reaches, opponent(player)),
      opponent_width, compiled.showdowns,
      showdown_values(river, buffers.values, player), player_width);

  update_player(river, board, player, buffers.reaches.workspaces[player_index],
                buffers.values, root_values, chance_reach, iteration_weight);
  terminals.mask_board_values(board, player, root_values);
}

void RiverTerminalOperator::evaluate_fold_board_batch(
    std::span<const BoardIndex> boards, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const float> payoffs, std::span<float> values,
    size_t evaluated_hand_stride) const {
  size_t board_count = boards.size();
  assert(opponent_reaches.size() ==
         payoffs.size() * board_count * opponent_hand_stride);
  assert(values.size() ==
         payoffs.size() * board_count * evaluated_hand_stride);
  std::vector<float> board_reaches(payoffs.size() * opponent_hand_stride);
  std::vector<float> board_values(payoffs.size() * evaluated_hand_stride);
  for (size_t board = 0; board < board_count; ++board) {
    for (size_t terminal = 0; terminal < payoffs.size(); ++terminal) {
      auto source = opponent_reaches.subspan(
          (terminal * board_count + board) * opponent_hand_stride,
          opponent_hand_stride);
      std::ranges::copy(
          source, board_reaches.begin() + terminal * opponent_hand_stride);
    }
    evaluate_folds(boards[board], evaluated_player, board_reaches,
                   opponent_hand_stride, payoffs, board_values,
                   evaluated_hand_stride);
    for (size_t terminal = 0; terminal < payoffs.size(); ++terminal) {
      auto source = std::span<const float>(board_values)
                        .subspan(terminal * evaluated_hand_stride,
                                 evaluated_hand_stride);
      std::ranges::copy(
          source, values.begin() +
                      (terminal * board_count + board) * evaluated_hand_stride);
    }
  }
}

void RiverTerminalOperator::evaluate_showdown_board_batch(
    std::span<const BoardIndex> boards, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const CompiledShowdown> showdowns, std::span<float> values,
    size_t evaluated_hand_stride) const {
  size_t board_count = boards.size();
  assert(opponent_reaches.size() ==
         showdowns.size() * board_count * opponent_hand_stride);
  assert(values.size() ==
         showdowns.size() * board_count * evaluated_hand_stride);
  std::vector<float> board_reaches(showdowns.size() * opponent_hand_stride);
  std::vector<float> board_values(showdowns.size() * evaluated_hand_stride);
  for (size_t board = 0; board < board_count; ++board) {
    for (size_t terminal = 0; terminal < showdowns.size(); ++terminal) {
      auto source = opponent_reaches.subspan(
          (terminal * board_count + board) * opponent_hand_stride,
          opponent_hand_stride);
      std::ranges::copy(
          source, board_reaches.begin() + terminal * opponent_hand_stride);
    }
    evaluate_showdowns(boards[board], evaluated_player, board_reaches,
                       opponent_hand_stride, showdowns, board_values,
                       evaluated_hand_stride);
    for (size_t terminal = 0; terminal < showdowns.size(); ++terminal) {
      auto source = std::span<const float>(board_values)
                        .subspan(terminal * evaluated_hand_stride,
                                 evaluated_hand_stride);
      std::ranges::copy(
          source, values.begin() +
                      (terminal * board_count + board) * evaluated_hand_stride);
    }
  }
}

void update_batch(StreetTree &river, std::span<const BatchJob> jobs,
                  Player player, const RiverTerminalOperator &terminals,
                  BatchContext &context, float iteration_weight) {
  const CompiledStreet &compiled = river.compiled;
  assert(compiled.street == Street::River);
  assert(compiled.boundary_value_count == 0);
  update_street_chain<Street::River, Street::River>(
      river, {}, jobs, player, terminals, context, iteration_weight, 0);
}

void update_turn(StreetTree &turn, std::span<StreetTree> future_street_trees,
                 std::span<const BatchJob> jobs, Player player,
                 const RiverTerminalOperator &terminals, BatchContext &context,
                 float iteration_weight, size_t boundary_batch_size) {
  const CompiledStreet &compiled = turn.compiled;
  assert(compiled.street == Street::Turn);
  update_street_chain<Street::Turn, Street::Turn>(
      turn, future_street_trees, jobs, player, terminals, context,
      iteration_weight, boundary_batch_size);
}

void update_chance_root(StreetTree &river, std::span<const BatchJob> jobs,
                        Player player, const RiverTerminalOperator &terminals,
                        BatchContext &context, std::span<float> root_values,
                        float iteration_weight) {
  assert(!jobs.empty());
  const CompiledStreet &compiled = river.compiled;
  size_t player_index = static_cast<size_t>(player);
  size_t opponent_index = static_cast<size_t>(opponent(player));
  size_t player_width = compiled.padded_hand_counts[player_index];
  assert(root_values.size() == player_width);

  auto &frame = context.frames[0];
  frame.child_jobs.resize(jobs.size());
  frame.player_root_masks.reserve(jobs.size() * player_width);
  frame.player_root_masks.resize(jobs.size() * player_width);
  frame.opponent_root_masks.reserve(
      jobs.size() * compiled.padded_hand_counts[opponent_index]);
  frame.opponent_root_masks.resize(
      jobs.size() * compiled.padded_hand_counts[opponent_index]);
  for (size_t board = 0; board < jobs.size(); ++board) {
    BatchJob &masked_job = frame.child_jobs[board];
    masked_job.board = jobs[board].board;
    masked_job.root_values = jobs[board].root_values;
    masked_job.chance_reach = jobs[board].chance_reach;

    std::span<float> player_masked =
        std::span<float>(frame.player_root_masks)
            .subspan(board * player_width, player_width);
    std::span<float> opponent_masked =
        std::span<float>(frame.opponent_root_masks)
            .subspan(board * compiled.padded_hand_counts[opponent_index],
                     compiled.padded_hand_counts[opponent_index]);
    terminals.mask_board_reaches(masked_job.board, player,
                                 jobs[board].root_reaches[player_index],
                                 player_masked);
    terminals.mask_board_reaches(masked_job.board, opponent(player),
                                 jobs[board].root_reaches[opponent_index],
                                 opponent_masked);
    masked_job.root_reaches[player_index] = player_masked;
    masked_job.root_reaches[opponent_index] = opponent_masked;
  }

  update_batch(river, frame.child_jobs, player, terminals, context,
               iteration_weight);
  std::ranges::fill(root_values, 0.0F);
  for (const BatchJob &job : frame.child_jobs) {
    assert(job.root_values.size() == player_width);
    for (size_t hand = 0; hand < player_width; ++hand) {
      root_values[hand] += job.chance_reach * job.root_values[hand];
    }
  }
}

} // namespace spmv_poker
