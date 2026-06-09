#pragma once

#include "spmv_poker/cards.h"
#include "spmv_poker/range.h"

#include <array>
#include <stdint.h>
#include <vector>

namespace spmv_poker {

constexpr size_t FINAL_BOARD_HAND_COUNT = 1081;

/**
 * Precomputed showdown ordering for every possible runout
 *
 * hand_table is the canonical list of hands that do not overlap the board.
 * The rest of the showdown results are computed using this canonical ordering
 *
 * for each runout i, ranked_hand_indices[i] contains the 1,081 legal hole-card
 * hand indices (based on the hand_table indexing), ordered from strongest to weakest.
 *
 * To avoid having to explicitly store the computed ranking of each hand, we store as
 * groups of "tied" hands. 
 *
 * group_boundaries separates each runout's ordered hands into these tie groups,
 * while runout_group_offsets identifies the boundaries belonging to each runout (a bit like CSR format)
 * When computing showdown vales, we sweep over the groups to accumulate weaker and stronger
 * opponent weight, excluding ties and hands blocked by the current hand.
 */
struct ShowdownTables {
  std::vector<Hand> hand_table;
  std::vector<std::array<uint8_t, 5>> runouts;
  std::vector<std::array<uint16_t, FINAL_BOARD_HAND_COUNT>> ranked_hand_indices;
  std::vector<uint16_t> group_boundaries;
  std::vector<uint32_t> runout_group_offsets;

  explicit ShowdownTables(const std::array<uint8_t, 3> &flop);
  explicit ShowdownTables(const std::array<uint8_t, 4> &turn);
  explicit ShowdownTables(const std::array<uint8_t, 5> &river);

  void apply_showdown(size_t runout_index, const Range &opponent_range,
                      std::vector<float> &showdown_values) const;

private:
  void build_hand_table(uint64_t dead_cards);
  void reserve_runouts(size_t runout_count);
  void add_runout(const std::array<uint8_t, 5> &board);
};

} // namespace spmv_poker
