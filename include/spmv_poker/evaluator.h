#pragma once

#include "spmv_poker/game_tree.h"
#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <vector>

namespace spmv_poker {

/**
 * Recursively evaluates the fixed strategy in a game tree
 *
 * values, opponent reach weights, and strategies use recursion-depth scratch
 */
struct RecursiveEvaluator {
  const GameTree &tree;
  const TerminalTables &terminals;
  std::vector<float> value_scratch;
  std::vector<float> reach_scratch;
  std::vector<float> strategy_scratch;
  size_t value_stride = 0;
  size_t reach_stride = 0;
  size_t strategy_stride = 0;

  RecursiveEvaluator(const GameTree &tree, const TerminalTables &terminals)
      : tree(tree), terminals(terminals) {}

  void evaluate(Player player, const Range &opponent_range,
                std::vector<float> &values);

private:
  [[nodiscard]] std::span<float> values(size_t depth);
  [[nodiscard]] std::span<float> reach(size_t depth);
  [[nodiscard]] std::span<float> strategy(size_t depth,
                                          const DecisionNode &node);
  void evaluate_node(NodeIndex node_index, Player player, size_t depth);
  [[nodiscard]] size_t maximum_depth(NodeIndex node_index) const;
};

} // namespace spmv_poker
