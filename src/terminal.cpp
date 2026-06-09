#include "spmv_poker/terminal.h"

#include <phevaluator/phevaluator.h>

#include <algorithm>
#include <array>
#include <utility>

namespace spmv_poker {

namespace {

struct EvaluatedHand {
  uint16_t hand_index;
  uint16_t rank;
};

} // namespace

TerminalTables::TerminalTables(const std::array<uint8_t, 3> &flop) {
  uint64_t dead_cards = make_mask(flop);
  build_hand_table(dead_cards);
  reserve_runouts((DECK_SIZE - flop.size()) * (DECK_SIZE - flop.size() - 1) /
                  2);

  for (size_t turn = 0; turn < DECK_SIZE; ++turn) {
    if ((dead_cards & card_mask(static_cast<uint8_t>(turn))) != 0) {
      continue;
    }
    for (size_t river = turn + 1; river < DECK_SIZE; ++river) {
      if ((dead_cards & card_mask(static_cast<uint8_t>(river))) == 0) {
        add_runout({flop[0], flop[1], flop[2], static_cast<uint8_t>(turn),
                    static_cast<uint8_t>(river)});
      }
    }
  }
}

TerminalTables::TerminalTables(const std::array<uint8_t, 4> &turn) {
  uint64_t dead_cards = make_mask(turn);
  build_hand_table(dead_cards);
  reserve_runouts(DECK_SIZE - turn.size());

  for (size_t river = 0; river < DECK_SIZE; ++river) {
    if ((dead_cards & card_mask(static_cast<uint8_t>(river))) == 0) {
      add_runout(
          {turn[0], turn[1], turn[2], turn[3], static_cast<uint8_t>(river)});
    }
  }
}

TerminalTables::TerminalTables(const std::array<uint8_t, 5> &river) {
  build_hand_table(make_mask(river));
  reserve_runouts(1);
  add_runout(river);
}

void TerminalTables::build_hand_table(uint64_t dead_cards) {
  hand_table.reserve(DECK_SIZE * (DECK_SIZE - 1) / 2);

  for (size_t first = 0; first < DECK_SIZE; ++first) {
    for (size_t second = first + 1; second < DECK_SIZE; ++second) {
      auto first_card = static_cast<uint8_t>(first);
      auto second_card = static_cast<uint8_t>(second);
      uint64_t mask = card_mask(first_card) | card_mask(second_card);
      if ((mask & dead_cards) == 0) {
        hand_table.push_back({first_card, second_card, mask});
      }
    }
  }
}

void TerminalTables::reserve_runouts(size_t runout_count) {
  runouts.reserve(runout_count);
  ranked_hand_indices.reserve(runout_count);
  runout_group_offsets.reserve(runout_count);
}

void TerminalTables::add_runout(const std::array<uint8_t, 5> &board) {
  uint64_t board_mask = make_mask(board);
  runouts.push_back(board);
  std::array<EvaluatedHand, FINAL_BOARD_HAND_COUNT> evaluated_hands;
  size_t ranked_hand_count = 0;

  // evaluate every canonical hand that does not overlap the final board
  for (size_t index = 0; index < hand_table.size(); ++index) {
    const Hand &hand = hand_table[index];
    if ((hand.mask & board_mask) != 0) {
      continue;
    }

    evaluated_hands[ranked_hand_count++] = {
        static_cast<uint16_t>(index),
        static_cast<uint16_t>(evaluate_7cards(board[0], board[1], board[2],
                                              board[3], board[4], hand.first,
                                              hand.second))};
  }

  std::sort(evaluated_hands.begin(), evaluated_hands.end(),
            [](const EvaluatedHand &first, const EvaluatedHand &second) {
              return first.rank < second.rank;
            });

  // keep only the sorted hand indices and the boundaries between ties
  std::array<uint16_t, FINAL_BOARD_HAND_COUNT> hand_indices;
  runout_group_offsets.push_back(
      static_cast<uint32_t>(group_boundaries.size()));
  group_boundaries.push_back(0);
  for (size_t index = 0; index < evaluated_hands.size(); ++index) {
    hand_indices[index] = evaluated_hands[index].hand_index;
    if (index > 0 &&
        evaluated_hands[index].rank != evaluated_hands[index - 1].rank) {
      group_boundaries.push_back(static_cast<uint16_t>(index));
    }
  }
  group_boundaries.push_back(static_cast<uint16_t>(hand_indices.size()));
  ranked_hand_indices.push_back(std::move(hand_indices));
}

void TerminalTables::apply_fold(const Range &opponent_range, float payoff,
                                std::vector<float> &values) const {
  float total = 0.0F;
  std::array<float, DECK_SIZE> card_totals{};

  for (size_t index = 0; index < hand_table.size(); ++index) {
    const Hand &hand = hand_table[index];
    float weight = opponent_range.weights[index];
    total += weight;
    card_totals[hand.first] += weight;
    card_totals[hand.second] += weight;
  }

  values.resize(hand_table.size());
  for (size_t index = 0; index < hand_table.size(); ++index) {
    const Hand &hand = hand_table[index];
    float compatible_weight =
        total - card_totals[hand.first] - card_totals[hand.second] +
        opponent_range.weights[index];
    values[index] = payoff * compatible_weight;
  }
}

void TerminalTables::apply_showdown(size_t runout_index,
                                    const Range &opponent_range,
                                    float win_payoff, float loss_payoff,
                                    std::vector<float> &values) const {
  size_t hand_count = hand_table.size();
  auto &hand_indices = ranked_hand_indices[runout_index];
  size_t boundary_begin = runout_group_offsets[runout_index];
  size_t boundary_end = runout_index + 1 < runout_group_offsets.size()
                            ? runout_group_offsets[runout_index + 1]
                            : group_boundaries.size();
  values.assign(hand_count, 0.0F);

  float total = 0.0F;
  std::array<float, DECK_SIZE> card_totals{};

  // accumulate compatible opponent weight beaten by each hand
  for (size_t boundary = boundary_end - 1; boundary-- > boundary_begin;) {
    size_t group_begin = group_boundaries[boundary];
    size_t group_end = group_boundaries[boundary + 1];
    for (size_t index = group_begin; index < group_end; ++index) {
      uint16_t hand_index = hand_indices[index];
      const Hand &hand = hand_table[hand_index];
      values[hand_index] =
          win_payoff *
          (total - card_totals[hand.first] - card_totals[hand.second]);
    }

    for (size_t index = group_begin; index < group_end; ++index) {
      uint16_t hand_index = hand_indices[index];
      const Hand &hand = hand_table[hand_index];
      float weight = opponent_range.weights[hand_index];
      total += weight;
      card_totals[hand.first] += weight;
      card_totals[hand.second] += weight;
    }
  }

  total = 0.0F;
  card_totals.fill(0.0F);

  // subtract opponent weight that beats each hand
  for (size_t boundary = boundary_begin; boundary + 1 < boundary_end;
       ++boundary) {
    size_t group_begin = group_boundaries[boundary];
    size_t group_end = group_boundaries[boundary + 1];
    for (size_t index = group_begin; index < group_end; ++index) {
      uint16_t hand_index = hand_indices[index];
      const Hand &hand = hand_table[hand_index];
      values[hand_index] +=
          loss_payoff *
          (total - card_totals[hand.first] - card_totals[hand.second]);
    }

    for (size_t index = group_begin; index < group_end; ++index) {
      uint16_t hand_index = hand_indices[index];
      const Hand &hand = hand_table[hand_index];
      float weight = opponent_range.weights[hand_index];
      total += weight;
      card_totals[hand.first] += weight;
      card_totals[hand.second] += weight;
    }
  }
}

} // namespace spmv_poker
