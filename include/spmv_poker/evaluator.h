#pragma once

#include "spmv_poker/game_tree.h"
#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <vector>

namespace spmv_poker {

/**
 * Recursively evaluates the fixed strategy in a game tree
 */
struct RecursiveEvaluator {
  GameTree &tree;
  const TerminalTables &terminals;

  RecursiveEvaluator(GameTree &tree, const TerminalTables &terminals)
      : tree(tree), terminals(terminals) {}

  void evaluate(Player player, const Range &opponent_range,
                std::vector<float> &values);
  void cfr_update(Player player, const Range &player_range,
                  const Range &opponent_range, std::vector<float> &values,
                  float iteration_weight = 1.0F);
  void cfr_iteration(const Range &hero_range, const Range &villain_range,
                     float iteration_weight = 1.0F);
  void best_response(Player player, const Range &opponent_range,
                     std::vector<float> &values,
                     bool use_average_strategy = true);
  [[nodiscard]] float best_response_value(Player player,
                                          const Range &player_range,
                                          const Range &opponent_range,
                                          bool use_average_strategy = true);
  [[nodiscard]] float exploitability(const Range &hero_range,
                                     const Range &villain_range,
                                     bool use_average_strategy = true);

private:
  void evaluate_node(NodeIndex node_index, Player player,
                     std::span<const float> opponent_range,
                     std::vector<float> &values, uint64_t board_mask);
  void cfr_update_node(NodeIndex node_index, Player player,
                       std::span<const float> player_reach,
                       std::span<const float> opponent_reach,
                       std::vector<float> &values, float chance_reach,
                       float iteration_weight, uint64_t board_mask);
  void best_response_node(NodeIndex node_index, Player player,
                          std::span<const float> opponent_reach,
                          std::vector<float> &values,
                          bool use_average_strategy, uint64_t board_mask);
  void terminal_fold(const GameNode &node, Player player,
                     std::span<const float> opponent_reach,
                     std::vector<float> &values, uint64_t board_mask) const;
  void terminal_showdown(const GameNode &node, Player player,
                         std::span<const float> opponent_reach,
                         std::vector<float> &values,
                         uint64_t board_mask) const;
  void decision_strategy(const DecisionNode &decision,
                         bool use_average_strategy,
                         std::vector<float> &strategy) const;
};

} // namespace spmv_poker
