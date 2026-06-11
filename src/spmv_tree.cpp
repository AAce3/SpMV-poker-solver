#include "spmv_poker/spmv_tree.h"

#include <array>
#include <bit>
#include <cassert>

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#endif

namespace spmv_poker {

namespace {

constexpr uint64_t DECK_MASK = (uint64_t{1} << DECK_SIZE) - 1;

uint8_t select_set_bit(uint64_t bits, BoardIndex index) {
  assert(index < std::popcount(bits));

  uint8_t result = 0;
  for (uint8_t width : std::array<uint8_t, 6>{32, 16, 8, 4, 2, 1}) {
    uint64_t lower = bits & ((uint64_t{1} << width) - 1);
    auto lower_count = static_cast<BoardIndex>(std::popcount(lower));
    if (index < lower_count) {
      bits = lower;
    } else {
      index -= lower_count;
      bits >>= width;
      result += width;
    }
  }
  return result;
}

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
[[gnu::target("bmi2")]]
uint8_t select_set_bit_bmi2(uint64_t bits, BoardIndex index) {
  return static_cast<uint8_t>(
      std::countr_zero(_pdep_u64(uint64_t{1} << index, bits)));
}

bool supports_bmi2() {
  static const bool supported = __builtin_cpu_supports("bmi2");
  return supported;
}
#endif

uint8_t select_available_card(uint64_t board_mask, BoardIndex index) {
  uint64_t available_cards = DECK_MASK & ~board_mask;
  assert(index < std::popcount(available_cards));
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  if (supports_bmi2()) {
    return select_set_bit_bmi2(available_cards, index);
  }
#endif
  return select_set_bit(available_cards, index);
}

} // namespace

NodeIndex StreetTopology::add_showdown_node(float win_payoff,
                                            float loss_payoff) {
  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  nodes.push_back(StreetNode{.type = StreetNodeType::Showdown,
                             .win_payoff = win_payoff,
                             .loss_payoff = loss_payoff});
  return index;
}

NodeIndex StreetTopology::add_boundary_node() {
  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  nodes.push_back(StreetNode{.type = StreetNodeType::StreetBoundary});
  return index;
}

NodeIndex StreetTopology::add_player_node(
    Player player, std::span<const NodeIndex> child_nodes,
    std::optional<float> fold_payoff) {
  assert(!child_nodes.empty() || fold_payoff.has_value());
  for (NodeIndex child : child_nodes) {
    assert(child < nodes.size());
  }

  NodeIndex index = static_cast<NodeIndex>(nodes.size());
  size_t child_begin = children.size();
  children.insert(children.end(), child_nodes.begin(), child_nodes.end());

  size_t hand_count = hand_count_for(player);
  size_t cfr_offset_within_board = state_entries_per_board;
  size_t action_count =
      child_nodes.size() + static_cast<size_t>(fold_payoff.has_value());
  state_entries_per_board += hand_count * action_count;

  nodes.push_back(StreetNode{
      .type = StreetNodeType::Player,
      .child_range = {child_begin, child_nodes.size()},
      .player = player,
      .cfr_offset_within_board = cfr_offset_within_board,
      .fold_payoff =
          fold_payoff.value_or(std::numeric_limits<float>::quiet_NaN()),
  });
  return index;
}

StreetGame::StreetGame(StreetTopology topology, uint64_t flop_mask)
    : topology(std::move(topology)), board_index(flop_mask),
      state{
          .regrets =
              std::vector<float>(board_index.board_count(this->topology.street) *
                                     this->topology.state_entries_per_board,
                                 0.0F),
          .strategy_sum =
              std::vector<float>(board_index.board_count(this->topology.street) *
                                     this->topology.state_entries_per_board,
                                 0.0F),
      } {}

size_t RunoutIndex::board_count(Street street) const {
  switch (street) {
  case Street::Flop:
    return 1;
  case Street::Turn:
    return TURN_BOARD_COUNT;
  case Street::River:
    return RIVER_BOARD_COUNT;
  }
  assert(false);
  return 0;
}

size_t RunoutIndex::child_count(Street street) const {
  assert(street != Street::River);
  return street == Street::Flop ? TURN_BOARD_COUNT : RIVER_CHILDREN_PER_TURN;
}

BoardIndex RunoutIndex::compressed_card_index(uint64_t board_mask,
                                              uint8_t card) {
  assert(card < DECK_SIZE);
  assert((board_mask & card_mask(card)) == 0);
  uint64_t lower_cards = card == 0 ? 0 : card_mask(card) - 1;
  return card -
         static_cast<BoardIndex>(std::popcount(board_mask & lower_cards));
}

uint8_t RunoutIndex::card_at_compressed_index(uint64_t board_mask,
                                              BoardIndex index) {
  // CUDA should implement this select operation with __fns.
  return select_available_card(board_mask, index);
}

BoardIndex RunoutIndex::child_board(Street street, BoardIndex parent_board,
                                    uint8_t dealt_card) const {
  assert(street != Street::River);
  assert(parent_board < board_count(street));
  uint64_t parent_mask = board_mask(street, parent_board);
  BoardIndex child = compressed_card_index(parent_mask, dealt_card);
  if (street == Street::Turn) {
    child += parent_board * RIVER_CHILDREN_PER_TURN;
  }
  return child;
}

uint8_t RunoutIndex::turn_card(BoardIndex turn_board) const {
  assert(turn_board < TURN_BOARD_COUNT);
  return card_at_compressed_index(flop_mask, turn_board);
}

uint8_t RunoutIndex::river_card(BoardIndex river_board) const {
  assert(river_board < RIVER_BOARD_COUNT);
  BoardIndex turn_board = river_board / RIVER_CHILDREN_PER_TURN;
  BoardIndex river_offset = river_board % RIVER_CHILDREN_PER_TURN;
  return card_at_compressed_index(flop_mask | card_mask(turn_card(turn_board)),
                                  river_offset);
}

uint64_t RunoutIndex::board_mask(Street street, BoardIndex board) const {
  assert(board < board_count(street));
  if (street == Street::Flop) {
    return flop_mask;
  }

  BoardIndex turn_board =
      street == Street::Turn ? board : board / RIVER_CHILDREN_PER_TURN;
  uint64_t mask = flop_mask | card_mask(turn_card(turn_board));
  if (street == Street::River) {
    mask |= card_mask(river_card(board));
  }
  return mask;
}

} // namespace spmv_poker
