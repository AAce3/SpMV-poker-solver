#pragma once

#include "spmv_poker/decision_node.h"

#include <stdint.h>
#include <span>
#include <vector>

namespace spmv_poker {

using NodeIndex = uint32_t;

constexpr uint32_t NO_DECISION = UINT32_MAX;

enum class NodeType : uint8_t {
  Decision,
  Chance,
  Fold,
  Showdown,
};

enum class Player : uint8_t {
  Hero,
  Villain,
};

struct GameEdge {
  NodeIndex child;
  float probability;
};

// nodes store indices into a global tree vector
struct GameNode {
  NodeType type;
  IndexRange edge_range{0, 0};
  // index of the decision node
  uint32_t decision_index = NO_DECISION;
  Player player = Player::Hero;
  uint32_t runout_index = 0;
  float payoff = 0.0F;
  float win_payoff = 0.0F;
  float loss_payoff = 0.0F;
};

struct GameTree {
  size_t hand_count;
  NodeIndex root = 0;
  std::vector<GameNode> nodes;
  std::vector<GameEdge> edges;
  std::vector<DecisionNode> decisions;
  SolverState state;

  explicit GameTree(size_t hand_count) : hand_count(hand_count) {}

  NodeIndex add_fold_node(float payoff);
  NodeIndex add_showdown_node(uint32_t runout_index, float win_payoff,
                              float loss_payoff);
  NodeIndex add_chance_node(std::span<const GameEdge> children);
  NodeIndex add_decision_node(Player player,
                              std::span<const NodeIndex> children);

  [[nodiscard]] std::span<const GameEdge> children(const GameNode &node) const {
    return node.edge_range.view(std::span(edges));
  }
};

} // namespace spmv_poker
