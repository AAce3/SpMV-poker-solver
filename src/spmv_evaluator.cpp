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

StreetBatchView OwnedStreetBatch::view() {
  return StreetBatchView{
      .boards = std::span<const BoardIndex>(boards),
      .root_reaches = {std::span<const float>(root_reaches[0]),
                       std::span<const float>(root_reaches[1])},
      .root_values = std::span<float>(root_values),
      .cumulative_chance_reaches =
          std::span<const float>(cumulative_chance_reaches),
  };
}

void validate_batch(const StreetBatchView &batch, const CompiledStreet &compiled,
                    Player evaluated_player) {
  size_t board_count = batch.board_count();
  for (Player player : {Player::Hero, Player::Villain}) {
    size_t player_index = static_cast<size_t>(player);
    size_t width = compiled.padded_hand_counts[player_index];
    assert(batch.root_reaches[player_index].size() == board_count * width);
  }

  if (!batch.root_values.empty()) {
    size_t evaluated_index = static_cast<size_t>(evaluated_player);
    size_t evaluated_width = compiled.padded_hand_counts[evaluated_index];
    assert(batch.root_values.size() == board_count * evaluated_width);
  }
  assert(batch.cumulative_chance_reaches.empty() ||
         batch.cumulative_chance_reaches.size() == board_count);
}

void propagate_reaches_batch(const StreetTree &game,
                             const StreetBatchView &batch,
                             std::array<std::span<float>, 2> reach_workspaces) {
  const CompiledStreet &compiled = game.compiled;
  validate_batch(batch, compiled, Player::Hero);
  size_t board_count = batch.board_count();

  for (Player player : {Player::Hero, Player::Villain}) {
    size_t player_index = static_cast<size_t>(player);
    size_t width = compiled.padded_hand_counts[player_index];
    const CompiledForwardPlan &plan = compiled.forward_plans[player_index];
    size_t required = static_cast<size_t>(plan.workspace_slot_count) *
                      board_count * width;
    assert(reach_workspaces[player_index].size() >= required);
    std::span<float> active_workspace =
        reach_workspaces[player_index].first(required);
    std::ranges::fill(active_workspace, 0.0F);

    SlotBoardHandView<float> workspace(active_workspace,
                                       plan.workspace_slot_count, board_count,
                                       width);
    BoardHandView<const float> roots(batch.root_reaches[player_index],
                                     board_count, width);

    for (size_t board = 0; board < board_count; ++board) {
      std::ranges::copy(
          roots.hands(board),
          workspace.hands(plan.root_input_reach_slot, board).begin());
    }

    for (const CompiledForwardNode &node : plan.nodes) {
      bool updates_player = node.player == player;
      float uniform_probability = 1.0F / static_cast<float>(node.action_count);

      for (size_t board = 0; board < board_count; ++board) {
        std::span<const float> board_regrets =
            game.board_regrets(batch.boards[board]);
        auto input = workspace.hands(node.input_reach_slot, board);

        for (size_t hand_begin = 0; hand_begin < width;
             hand_begin += HAND_BLOCK_SIZE) {
          auto input_block = input.subspan(hand_begin, HAND_BLOCK_SIZE);
          if (updates_player) {
            alignas(64) std::array<float, HAND_BLOCK_SIZE>
                inverse_positive_sums{};
            compute_regret_matching_factors(
                board_regrets, node.cfr_offset_within_board, node.action_begin,
                node.action_count, hand_begin,
                std::span<const CompiledForwardAction>(plan.actions),
                std::span<float, HAND_BLOCK_SIZE>(inverse_positive_sums));
            for (size_t action = 0; action < node.action_count; ++action) {
              const CompiledForwardAction &compiled_action =
                  plan.actions[node.action_begin + action];
              auto destination =
                  workspace.hands(compiled_action.destination_slot, board)
                      .subspan(hand_begin, HAND_BLOCK_SIZE);
              const float *regrets = board_regrets.data() +
                                     node.cfr_offset_within_board +
                                     compiled_action.cfr_action_offset +
                                     hand_begin;
              for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
                float inverse_positive_sum = inverse_positive_sums[lane];
                float strategy = inverse_positive_sum > 0.0F
                                     ? std::max(0.0F, regrets[lane]) *
                                           inverse_positive_sum
                                     : uniform_probability;
                destination[lane] += input_block[lane] * strategy;
              }
            }
          } else {
            for (size_t action = 0; action < node.action_count; ++action) {
              const CompiledForwardAction &compiled_action =
                  plan.actions[node.action_begin + action];
              auto destination =
                  workspace.hands(compiled_action.destination_slot, board)
                      .subspan(hand_begin, HAND_BLOCK_SIZE);
              for (size_t lane = 0; lane < HAND_BLOCK_SIZE; ++lane) {
                destination[lane] += input_block[lane];
              }
            }
          }
        }
      }
    }
  }
}

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

StreetTree make_tree_from_state(const CompiledStreet &compiled,
                                const RunoutIndex &board_index,
                                std::optional<StreetState> &state_slot) {
  StreetState state;
  if (state_slot) {
    state = std::move(*state_slot);
  }

  size_t board_count = board_index.board_count(compiled.street);
  size_t expected_entries =
      board_count * static_cast<size_t>(compiled.state_entries_per_board);
  if (state.regrets.size() != expected_entries) {
    state.regrets.assign(expected_entries, 0.0F);
  }
  if (state.cumulative_strategy.size() != expected_entries) {
    state.cumulative_strategy.assign(expected_entries, 0.0F);
  }

  return StreetTree(compiled, board_index, std::move(state.regrets),
                    std::move(state.cumulative_strategy));
}

StreetTree make_tree_from_state(const CompiledStreet &compiled,
                                const RunoutIndex &board_index,
                                StreetState &state_slot) {
  size_t board_count = board_index.board_count(compiled.street);
  size_t expected_entries =
      board_count * static_cast<size_t>(compiled.state_entries_per_board);
  if (state_slot.regrets.size() != expected_entries) {
    state_slot.regrets.assign(expected_entries, 0.0F);
  }
  if (state_slot.cumulative_strategy.size() != expected_entries) {
    state_slot.cumulative_strategy.assign(expected_entries, 0.0F);
  }

  return StreetTree(compiled, board_index, std::move(state_slot.regrets),
                    std::move(state_slot.cumulative_strategy));
}

void store_tree_state(std::optional<StreetState> &state_slot,
                      StreetTree &tree) {
  state_slot = StreetState{
      .regrets = std::move(tree.regrets),
      .cumulative_strategy = std::move(tree.cumulative_strategy),
  };
}

void store_tree_state(StreetState &state_slot, StreetTree &tree) {
  state_slot.regrets = std::move(tree.regrets);
  state_slot.cumulative_strategy = std::move(tree.cumulative_strategy);
}

void CpuSolveExecutor::update_player(const SolveProgram &program,
                                     SolveState &state,
                                     CpuSolveWorkspace &workspace,
                                     Player player, RootRangesView root_ranges,
                                     std::span<float> root_values,
                                     float iteration_weight) {
  assert(program.board_index.has_value());
  assert(program.terminals != nullptr);
  RunoutIndex board_index = *program.board_index;
  const RiverTerminalOperator &terminals = *program.terminals;
  size_t player_index = static_cast<size_t>(player);
  size_t opponent_index = static_cast<size_t>(opponent(player));

  StreetTree flop_tree = program.flop.has_value()
                             ? make_tree_from_state(*program.flop, board_index,
                                                    state.flop)
                             : StreetTree(program.river, board_index,
                                          std::vector<float>(),
                                          std::vector<float>());
  StreetTree turn_tree = program.turn.has_value()
                             ? make_tree_from_state(*program.turn, board_index,
                                                    state.turn)
                             : StreetTree(program.river, board_index,
                                          std::vector<float>(),
                                          std::vector<float>());
  StreetTree river_tree =
      make_tree_from_state(program.river, board_index, state.river);

  auto update_terminal_values = [&](const StreetTree &street,
                                    BoardIndex board,
                                    StreetReachBuffers &reaches,
                                    StreetValueBuffer &values) {
    prepare_value_buffer(street, player, values);
    size_t player_index = static_cast<size_t>(player);
    size_t opponent_index = static_cast<size_t>(opponent(player));
    size_t player_width =
        street.compiled.padded_hand_counts[player_index];
    size_t opponent_width =
        street.compiled.padded_hand_counts[opponent_index];
    terminals.evaluate_folds(
        board, player,
        fold_reaches(street, reaches, opponent(player)), opponent_width,
        street.compiled.fold_payoffs, fold_values(street, values, player),
        player_width);
    if (street.compiled.street == Street::River) {
      terminals.evaluate_showdowns(
          board, player,
          showdown_reaches(street, reaches, opponent(player)), opponent_width,
          street.compiled.showdowns, showdown_values(street, values, player),
          player_width);
    } else {
      terminals.evaluate_showdowns(
          board, player,
          showdown_reaches(street, reaches, opponent(player)), opponent_width,
          street.compiled.showdowns, showdown_values(street, values, player),
          player_width);
    }
  };

  auto schedule_turn_groups = program.schedule.turn_groups;
  if (program.flop.has_value() && program.turn.has_value()) {
    assert(program.flop_transition_graph.has_value());
    assert(program.turn_transition_graph.has_value());
    const CompiledTransitionGraph &flop_to_turn =
        *program.flop_transition_graph;
    const CompiledTransitionGraph &turn_to_river =
        *program.turn_transition_graph;

    StreetReachBuffers flop_reaches;
    propagate_reaches(flop_tree, 0, root_ranges, flop_reaches);
    StreetValueBuffer flop_values;
    prepare_value_buffer(flop_tree, player, flop_values);
    update_terminal_values(flop_tree, 0, flop_reaches, flop_values);

    size_t flop_player_index = static_cast<size_t>(player);
    size_t flop_opponent_index = static_cast<size_t>(opponent(player));
    size_t flop_player_width =
        flop_tree.compiled.padded_hand_counts[flop_player_index];
    size_t flop_opponent_width =
        flop_tree.compiled.padded_hand_counts[flop_opponent_index];
    std::span<const float> flop_player_boundary =
        boundary_reaches(flop_tree, flop_reaches, player);
    std::span<const float> flop_opponent_boundary =
        boundary_reaches(flop_tree, flop_reaches, opponent(player));

    for (const TurnGroup &turn_group : schedule_turn_groups) {
      for (uint32_t turn_offset = 0;
           turn_offset < turn_group.parent_board_count; ++turn_offset) {
        BoardIndex turn_board = turn_group.parent_board_begin + turn_offset;
        size_t turn_child_begin = flop_to_turn.child_offsets[turn_board];
        size_t turn_child_end = flop_to_turn.child_offsets[turn_board + 1];
        float turn_local_chance =
            flop_to_turn.local_chance_weights[turn_child_begin];

        StreetValueBuffer turn_values;
        StreetReachBuffers turn_reaches;
        std::array<std::span<const float>, 2> turn_roots{};

        for (size_t endpoint = 0;
             endpoint < flop_tree.compiled.boundary_value_count; ++endpoint) {
          size_t boundary_index = endpoint * 1;
          std::span<const float> player_boundary =
              flop_player_boundary.subspan(boundary_index * flop_player_width,
                                           flop_player_width);
          std::span<const float> opponent_boundary =
              flop_opponent_boundary.subspan(boundary_index *
                                                 flop_opponent_width,
                                             flop_opponent_width);

          std::vector<float> masked_player_root(flop_player_width, 0.0F);
          std::vector<float> masked_opponent_root(flop_opponent_width, 0.0F);
          terminals.mask_board_reaches(turn_board, player, player_boundary,
                                       masked_player_root);
          terminals.mask_board_reaches(turn_board, opponent(player),
                                       opponent_boundary,
                                       masked_opponent_root);
          turn_roots[player_index] = masked_player_root;
          turn_roots[opponent_index] = masked_opponent_root;
        }

        propagate_reaches(turn_tree, turn_board, turn_roots, turn_reaches);
        update_terminal_values(turn_tree, turn_board, turn_reaches, turn_values);

        std::span<const float> turn_player_boundary =
            boundary_reaches(turn_tree, turn_reaches, player);
        std::span<const float> turn_opponent_boundary =
            boundary_reaches(turn_tree, turn_reaches, opponent(player));

        std::vector<float> river_root_values(
            turn_tree.compiled.padded_hand_counts[player_index], 0.0F);

        for (BoardIndex river_board = turn_child_begin;
             river_board < turn_child_end; ++river_board) {
          float river_local_chance =
              turn_to_river.local_chance_weights[
                  turn_to_river.child_offsets[river_board]];
          float cumulative_chance =
              turn_local_chance * river_local_chance;
          std::array<std::span<const float>, 2> river_roots{};
          std::vector<float> masked_player_root(
              river_tree.compiled.padded_hand_counts[player_index], 0.0F);
          std::vector<float> masked_opponent_root(
              river_tree.compiled.padded_hand_counts[opponent_index], 0.0F);
          for (size_t endpoint = 0;
               endpoint < turn_tree.compiled.boundary_value_count; ++endpoint) {
            std::span<const float> player_boundary =
                turn_player_boundary.subspan(endpoint * flop_player_width,
                                             flop_player_width);
            std::span<const float> opponent_boundary =
                turn_opponent_boundary.subspan(endpoint * flop_opponent_width,
                                               flop_opponent_width);
            terminals.mask_board_reaches(river_board, player, player_boundary,
                                         masked_player_root);
            terminals.mask_board_reaches(river_board, opponent(player),
                                         opponent_boundary,
                                         masked_opponent_root);
          }
          river_roots[player_index] = masked_player_root;
          river_roots[opponent_index] = masked_opponent_root;

          RiverUpdateBuffers river_buffers;
          std::vector<float> child_root_values(
              river_tree.compiled.padded_hand_counts[player_index], 0.0F);
          update_river(river_tree, river_board, river_roots, player, terminals,
                       river_buffers, child_root_values, cumulative_chance,
                       iteration_weight);
          for (size_t hand = 0; hand < child_root_values.size(); ++hand) {
            river_root_values[hand] +=
                river_local_chance * child_root_values[hand];
          }
        }

        std::copy(river_root_values.begin(), river_root_values.end(),
                  turn_values.workspace.begin() +
                      static_cast<size_t>(turn_tree.compiled.boundary_value_begin) *
                      turn_tree.compiled.padded_hand_counts[player_index]);

        std::vector<float> turn_root_values(
            turn_tree.compiled.padded_hand_counts[player_index], 0.0F);
        ::spmv_poker::update_player(
            turn_tree, turn_board, player, turn_reaches.workspaces[player_index],
            turn_values, turn_root_values, turn_local_chance, iteration_weight);

        for (size_t hand = 0; hand < turn_root_values.size(); ++hand) {
          flop_values.workspace[flop_tree.compiled.boundary_value_begin *
                                    flop_player_width +
                                hand] +=
              turn_local_chance * turn_root_values[hand];
        }
      }
    }

    std::vector<float> flop_root_values(
        flop_tree.compiled.padded_hand_counts[player_index], 0.0F);
    ::spmv_poker::update_player(flop_tree, 0, player,
                                flop_reaches.workspaces[player_index],
                                flop_values, flop_root_values, 1.0F,
                                iteration_weight);
    std::copy(flop_root_values.begin(), flop_root_values.end(),
              root_values.begin());
    store_tree_state(state.flop, flop_tree);
    store_tree_state(state.turn, turn_tree);
    store_tree_state(state.river, river_tree);
    return;
  }

  std::array<std::span<const float>, 2> river_roots = root_ranges;
  RiverUpdateBuffers buffers;
  update_river(river_tree, 0, river_roots, player, terminals, buffers,
               root_values, 1.0F, iteration_weight);
  store_tree_state(state.river, river_tree);
}

} // namespace spmv_poker
