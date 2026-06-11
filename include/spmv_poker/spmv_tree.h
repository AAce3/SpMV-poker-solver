#pragma once

#include "spmv_poker/decision_node.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdint.h>
#include <span>
#include <utility>
#include <vector>

namespace spmv_poker {

constexpr size_t NO_CFR_STATE = SIZE_MAX;
constexpr size_t FLOP_CARD_COUNT = 3;
constexpr size_t TURN_BOARD_COUNT = DECK_SIZE - FLOP_CARD_COUNT;
constexpr size_t RIVER_CHILDREN_PER_TURN = TURN_BOARD_COUNT - 1;
constexpr size_t RIVER_BOARD_COUNT =
    TURN_BOARD_COUNT * RIVER_CHILDREN_PER_TURN;

enum class Street : uint8_t {
  Flop,
  Turn,
  River,
};

enum class StreetNodeType : uint8_t {
  Player,
  Showdown,
  StreetBoundary,
};

struct StreetNode {
  StreetNodeType type;
  IndexRange child_range{0, 0};
  Player player = Player::Hero;
  // Start of this node's [hand][action] entries within each board's CFR block.
  size_t cfr_offset_within_board = NO_CFR_STATE;
  float fold_payoff = std::numeric_limits<float>::quiet_NaN();
  float win_payoff = 0.0F;
  float loss_payoff = 0.0F;

  [[nodiscard]] bool has_fold() const {
    return !std::isnan(fold_payoff);
  }

  [[nodiscard]] size_t action_count() const {
    return child_range.count + static_cast<size_t>(has_fold());
  }

  [[nodiscard]] size_t continuing_action_begin() const {
    return static_cast<size_t>(has_fold());
  }

  [[nodiscard]] bool is_fold_action(size_t action) const {
    assert(action < action_count());
    return has_fold() && action == 0;
  }

  [[nodiscard]] bool has_state() const {
    return cfr_offset_within_board != NO_CFR_STATE;
  }
};

struct StreetTopology {
  Street street;
  std::array<size_t, 2> player_hand_counts;
  NodeIndex root = 0;
  std::vector<StreetNode> nodes;
  std::vector<NodeIndex> children;
  size_t state_entries_per_board = 0;

  StreetTopology(Street street, size_t hand_count)
      : street(street), player_hand_counts{hand_count, hand_count} {}
  StreetTopology(Street street, size_t hero_hand_count,
                 size_t villain_hand_count)
      : street(street),
        player_hand_counts{hero_hand_count, villain_hand_count} {}

  NodeIndex add_showdown_node(float win_payoff, float loss_payoff);
  NodeIndex add_boundary_node();
  NodeIndex add_player_node(
      Player player, std::span<const NodeIndex> child_nodes,
      std::optional<float> fold_payoff = std::nullopt);

  [[nodiscard]] std::span<const NodeIndex>
  child_nodes(const StreetNode &node) const {
    return node.child_range.view(children);
  }

  [[nodiscard]] NodeIndex child_for_action(const StreetNode &node,
                                           size_t action) const {
    assert(node.type == StreetNodeType::Player);
    assert(action < node.action_count());
    assert(!node.is_fold_action(action));
    return child_nodes(node)[action - node.continuing_action_begin()];
  }

  [[nodiscard]] size_t hand_count_for(Player player) const {
    return player_hand_counts[static_cast<size_t>(player)];
  }

  [[nodiscard]] size_t state_entry(const StreetNode &node, size_t hand,
                                   size_t action) const {
    assert(node.type == StreetNodeType::Player);
    assert(hand < hand_count_for(node.player));
    assert(action < node.action_count());
    return node.cfr_offset_within_board + hand * node.action_count() + action;
  }
};

// Direct board indexing for a solve rooted at one fixed flop.
struct RunoutIndex {

  explicit RunoutIndex(uint64_t flop_mask) : flop_mask(flop_mask) {
    assert(std::popcount(flop_mask) == FLOP_CARD_COUNT);
  }

  [[nodiscard]] size_t board_count(Street street) const;
  [[nodiscard]] size_t child_count(Street street) const;
  [[nodiscard]] BoardIndex child_board(Street street, BoardIndex parent_board,
                                       uint8_t dealt_card) const;
  [[nodiscard]] uint64_t board_mask(Street street, BoardIndex board) const;
  [[nodiscard]] uint8_t turn_card(BoardIndex turn_board) const;
  [[nodiscard]] uint8_t river_card(BoardIndex river_board) const;

private:
  uint64_t flop_mask;
  [[nodiscard]] static BoardIndex compressed_card_index(uint64_t board_mask,
                                                        uint8_t card);
  [[nodiscard]] static uint8_t card_at_compressed_index(uint64_t board_mask,
                                                        BoardIndex index);
};

struct StreetGame {
  const StreetTopology topology;
  const RunoutIndex board_index;
  SolverState state;

  StreetGame(StreetTopology topology, uint64_t flop_mask);

  [[nodiscard]] size_t state_entry(BoardIndex board,
                                   NodeIndex node_index, size_t hand,
                                   size_t action) const {
    assert(board < board_index.board_count(topology.street));
    assert(node_index < topology.nodes.size());
    return static_cast<size_t>(board) * topology.state_entries_per_board +
           topology.state_entry(topology.nodes[node_index], hand, action);
  }

  [[nodiscard]] std::span<float> board_regrets(BoardIndex board) {
    assert(board < board_index.board_count(topology.street));
    return std::span(state.regrets).subspan(
        static_cast<size_t>(board) * topology.state_entries_per_board,
        topology.state_entries_per_board);
  }

  [[nodiscard]] std::span<float> board_strategy_sum(BoardIndex board) {
    assert(board < board_index.board_count(topology.street));
    return std::span(state.strategy_sum)
        .subspan(static_cast<size_t>(board) * topology.state_entries_per_board,
                 topology.state_entries_per_board);
  }
};

} // namespace spmv_poker
