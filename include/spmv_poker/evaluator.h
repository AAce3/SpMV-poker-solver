#pragma once

#include "spmv_poker/game_tree.h"
#include "spmv_poker/range.h"
#include "spmv_poker/terminal.h"

#include <vector>

namespace spmv_poker {

/**
 * Evaluates the fixed strategy in a game tree.
 *
 * Terminal payoffs are stored from Hero's perspective. evaluate() returns one
 * counterfactual value per private hand for the selected player.
 */
struct Evaluator {
  const GameTree &tree;
  const TerminalTables &terminals;

  void evaluate(Player player, const Range &opponent_range,
                std::vector<float> &values) const;

private:
  void evaluate_node(NodeIndex node_index, Player player,
                     const Range &opponent_range,
                     std::vector<float> &values) const;
};

} // namespace spmv_poker
