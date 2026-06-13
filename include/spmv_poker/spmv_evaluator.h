#pragma once

#include "spmv_poker/tree_compiler.h"

#include <array>
#include <span>
#include <vector>

namespace spmv_poker {

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

struct RiverTerminalOperator {
  virtual ~RiverTerminalOperator() = default;

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

} // namespace spmv_poker
