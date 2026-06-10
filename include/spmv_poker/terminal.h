#pragma once

#include "spmv_poker/cards.h"
#include "spmv_poker/range.h"

#include <array>
#include <stdint.h>
#include <span>
#include <vector>

namespace spmv_poker {

struct RankGroup {
  uint16_t rank;
  uint16_t end;
};

struct RankedHandTable {
  std::vector<uint16_t> hand_indices;
  std::vector<RankGroup> groups;
  std::vector<IndexRange> runout_hand_ranges;
  std::vector<IndexRange> runout_group_ranges;
};

/**
 * Precomputed terminal rankings for two player-local hand tables.
 *
 * Each runout contains one separately sorted ranking for each player. Showdown
 * evaluation merges those rankings by poker rank.
 */
struct TerminalTables {
  std::array<std::vector<Hand>, 2> hand_tables;
  std::array<RankedHandTable, 2> ranked;
  std::vector<std::array<uint8_t, 5>> runouts;

  explicit TerminalTables(const std::array<uint8_t, 3> &flop);
  explicit TerminalTables(const std::array<uint8_t, 4> &turn);
  explicit TerminalTables(const std::array<uint8_t, 5> &river);
  TerminalTables(const std::array<uint8_t, 3> &flop, const Range &hero,
                 const Range &villain);
  TerminalTables(const std::array<uint8_t, 4> &turn, const Range &hero,
                 const Range &villain);
  TerminalTables(const std::array<uint8_t, 5> &river, const Range &hero,
                 const Range &villain);

  [[nodiscard]] std::span<const Hand> hands(Player player) const {
    return hand_tables[static_cast<size_t>(player)];
  }

  void apply_fold(Player evaluated_player, const Range &opponent_range,
                  float payoff, std::vector<float> &values) const;
  void apply_fold(Player evaluated_player,
                  std::span<const float> opponent_range, float payoff,
                  std::span<float> values) const;

  void apply_showdown(Player evaluated_player, size_t runout_index,
                      const Range &opponent_range, float win_payoff,
                      float loss_payoff, std::vector<float> &values) const;
  void apply_showdown(Player evaluated_player, size_t runout_index,
                      std::span<const float> opponent_range, float win_payoff,
                      float loss_payoff, std::span<float> values) const;

private:
  void set_full_hand_tables(uint64_t dead_cards);
  void set_hand_tables(uint64_t dead_cards, const Range &hero,
                       const Range &villain);
  void reserve_runouts(size_t runout_count);
  void add_runout(const std::array<uint8_t, 5> &board);
  void add_ranked_runout(Player player, const std::array<uint8_t, 5> &board);
};

} // namespace spmv_poker
