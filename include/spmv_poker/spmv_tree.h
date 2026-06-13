#pragma once

#include "spmv_poker/decision_node.h"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdint.h>
#include <utility>
#include <vector>

namespace spmv_poker {

constexpr size_t NO_CFR_STATE = SIZE_MAX;
constexpr size_t INVALID_INDEX = SIZE_MAX;
constexpr size_t FLOP_CARD_COUNT = 3;
constexpr size_t TURN_BOARD_COUNT = DECK_SIZE - FLOP_CARD_COUNT;
constexpr size_t RIVER_CHILDREN_PER_TURN = TURN_BOARD_COUNT - 1;
constexpr size_t RIVER_BOARD_COUNT = TURN_BOARD_COUNT * RIVER_CHILDREN_PER_TURN;
constexpr size_t HAND_BLOCK_SIZE = 32;

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
  // Start of this node's [action][padded hand] CFR block within one board
  size_t cfr_offset_within_board = NO_CFR_STATE;
  size_t padded_hand_count = 0;
  // Identifies this node's implicit fold in the shared topology float array
  size_t fold_index = INVALID_INDEX;
  // Identifies this node's reach in reach array
  size_t reach_index = INVALID_INDEX;
  float win_payoff = 0.0F;
  float loss_payoff = 0.0F;

  [[nodiscard]] bool has_fold() const { return fold_index != INVALID_INDEX; }

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

  [[nodiscard]] size_t state_entry_count() const {
    assert(type == StreetNodeType::Player);
    return action_count() * padded_hand_count;
  }

  [[nodiscard]] size_t action_entry(size_t action, size_t hand = 0) const {
    assert(type == StreetNodeType::Player);
    assert(action < action_count());
    assert(hand < padded_hand_count);
    return action * padded_hand_count + hand;
  }

  [[nodiscard]] size_t state_entry(size_t hand, size_t action) const {
    return cfr_offset_within_board + action_entry(action, hand);
  }

  [[nodiscard]] std::span<float>
  state_span(std::span<float> board_state) const {
    assert(has_state());
    return board_state.subspan(cfr_offset_within_board, state_entry_count());
  }

  [[nodiscard]] std::span<const float>
  state_span(std::span<const float> board_state) const {
    assert(has_state());
    return board_state.subspan(cfr_offset_within_board, state_entry_count());
  }
};

/*
 * Describes the betting decisions that can occur on one street, independently
 * of the concrete public board and private hands being evaluated
 *
 * Every public board on the street shares this topology because its available
 * checks, bets, calls, folds, and transitions have the same structure. Regrets
 * and cumulative strategy are not stored here
 */
struct StreetTopology {
  Street street;
  std::array<size_t, 2> player_hand_counts;
  NodeIndex root = 0;
  std::vector<StreetNode> nodes;
  std::vector<NodeIndex> children;
  std::vector<size_t> child_endpoints;
  std::vector<NodeIndex> endpoint_nodes;
  std::vector<float> fold_payoffs;
  std::array<size_t, 2> player_reach_counts{0, 0};
  size_t state_entries_per_board = 0;
  size_t showdown_count = 0;
  size_t boundary_count = 0;
  size_t endpoint_count = 0;

  StreetTopology(Street street, size_t hand_count)
      : street(street), player_hand_counts{hand_count, hand_count} {}

  StreetTopology(Street street, size_t hero_hand_count,
                 size_t villain_hand_count)
      : street(street),
        player_hand_counts{hero_hand_count, villain_hand_count} {}

  // Construction order is [showdowns][boundaries][child-before-parent players].
  NodeIndex add_showdown_node(float win_payoff, float loss_payoff);
  NodeIndex add_boundary_node();
  NodeIndex add_player_node(Player player,
                            std::span<const NodeIndex> child_nodes,
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

  [[nodiscard]] size_t endpoint_for_action(const StreetNode &node,
                                           size_t action) const {
    assert(node.type == StreetNodeType::Player);
    assert(action < node.action_count());
    assert(!node.is_fold_action(action));
    size_t child_offset =
        node.child_range.begin + action - node.continuing_action_begin();
    assert(!is_player(children[child_offset]));
    return child_endpoints[child_offset];
  }

  [[nodiscard]] NodeIndex boundary_begin() const {
    return static_cast<NodeIndex>(showdown_count);
  }

  [[nodiscard]] NodeIndex player_begin() const {
    return static_cast<NodeIndex>(showdown_count + boundary_count);
  }

  [[nodiscard]] bool is_showdown(NodeIndex node) const {
    return node < boundary_begin();
  }

  [[nodiscard]] bool is_boundary(NodeIndex node) const {
    return node >= boundary_begin() && node < player_begin();
  }

  [[nodiscard]] bool is_player(NodeIndex node) const {
    return node >= player_begin() && node < nodes.size();
  }

  [[nodiscard]] std::span<const StreetNode> showdowns() const {
    return std::span(nodes).first(showdown_count);
  }

  [[nodiscard]] std::span<const StreetNode> boundaries() const {
    return std::span(nodes).subspan(boundary_begin(), boundary_count);
  }

  [[nodiscard]] std::span<const StreetNode> players() const {
    return std::span(nodes).subspan(player_begin());
  }

  [[nodiscard]] float fold_payoff(const StreetNode &node) const {
    assert(node.has_fold());
    return fold_payoffs[node.fold_index];
  }

  [[nodiscard]] size_t hand_count_for(Player player) const {
    return player_hand_counts[static_cast<size_t>(player)];
  }

  [[nodiscard]] size_t padded_hand_count_for(Player player) const {
    size_t hand_count = hand_count_for(player);
    return (hand_count + HAND_BLOCK_SIZE - 1) / HAND_BLOCK_SIZE *
           HAND_BLOCK_SIZE;
  }
};

// board indexing for a solve rooted at one fixed flop
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
};

} // namespace spmv_poker
