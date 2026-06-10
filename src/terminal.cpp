#include "spmv_poker/terminal.h"

#include <phevaluator/phevaluator.h>

#include <algorithm>
#include <array>
#include <cassert>

namespace spmv_poker {

namespace {

struct EvaluatedHand {
  uint16_t hand_index;
  uint16_t rank;
};

constexpr size_t hand_key(const Hand &hand) {
  return static_cast<size_t>(hand.first) * DECK_SIZE + hand.second;
}

} // namespace

TerminalTables::TerminalTables(const std::array<uint8_t, 3> &flop) {
  uint64_t dead_cards = make_mask(flop);
  set_full_hand_tables(dead_cards);
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
  set_full_hand_tables(dead_cards);
  reserve_runouts(DECK_SIZE - turn.size());
  for (size_t river = 0; river < DECK_SIZE; ++river) {
    if ((dead_cards & card_mask(static_cast<uint8_t>(river))) == 0) {
      add_runout(
          {turn[0], turn[1], turn[2], turn[3], static_cast<uint8_t>(river)});
    }
  }
}

TerminalTables::TerminalTables(const std::array<uint8_t, 5> &river) {
  set_full_hand_tables(make_mask(river));
  reserve_runouts(1);
  add_runout(river);
}

TerminalTables::TerminalTables(const std::array<uint8_t, 3> &flop,
                               const Range &hero, const Range &villain) {
  uint64_t dead_cards = make_mask(flop);
  set_hand_tables(dead_cards, hero, villain);
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

TerminalTables::TerminalTables(const std::array<uint8_t, 4> &turn,
                               const Range &hero, const Range &villain) {
  uint64_t dead_cards = make_mask(turn);
  set_hand_tables(dead_cards, hero, villain);
  reserve_runouts(DECK_SIZE - turn.size());
  for (size_t river = 0; river < DECK_SIZE; ++river) {
    if ((dead_cards & card_mask(static_cast<uint8_t>(river))) == 0) {
      add_runout(
          {turn[0], turn[1], turn[2], turn[3], static_cast<uint8_t>(river)});
    }
  }
}

TerminalTables::TerminalTables(const std::array<uint8_t, 5> &river,
                               const Range &hero, const Range &villain) {
  set_hand_tables(make_mask(river), hero, villain);
  reserve_runouts(1);
  add_runout(river);
}

void TerminalTables::set_full_hand_tables(uint64_t dead_cards) {
  for (size_t first = 0; first < DECK_SIZE; ++first) {
    for (size_t second = first + 1; second < DECK_SIZE; ++second) {
      auto first_card = static_cast<uint8_t>(first);
      auto second_card = static_cast<uint8_t>(second);
      Hand hand{first_card, second_card,
                card_mask(first_card) | card_mask(second_card)};
      if ((hand.mask & dead_cards) == 0) {
        hand_tables[0].push_back(hand);
      }
    }
  }
  hand_tables[1] = hand_tables[0];
}

void TerminalTables::set_hand_tables(uint64_t dead_cards, const Range &hero,
                                     const Range &villain) {
  const std::array ranges{&hero, &villain};
  for (size_t player = 0; player < ranges.size(); ++player) {
    assert(ranges[player]->hands.size() == ranges[player]->weights.size());
    hand_tables[player] = ranges[player]->hands;
    for (const Hand &hand : hand_tables[player]) {
      assert((hand.mask & dead_cards) == 0);
    }
  }
}

void TerminalTables::reserve_runouts(size_t runout_count) {
  runouts.reserve(runout_count);
  for (RankedHandTable &table : ranked) {
    table.runout_hand_ranges.reserve(runout_count);
    table.runout_group_ranges.reserve(runout_count);
  }
}

void TerminalTables::add_runout(const std::array<uint8_t, 5> &board) {
  runouts.push_back(board);
  add_ranked_runout(Player::Hero, board);
  add_ranked_runout(Player::Villain, board);
}

void TerminalTables::add_ranked_runout(Player player,
                                       const std::array<uint8_t, 5> &board) {
  size_t player_index = static_cast<size_t>(player);
  const auto &hands = hand_tables[player_index];
  RankedHandTable &table = ranked[player_index];
  uint64_t board_mask = make_mask(board);
  std::vector<EvaluatedHand> evaluated_hands;
  evaluated_hands.reserve(hands.size());

  for (size_t index = 0; index < hands.size(); ++index) {
    const Hand &hand = hands[index];
    if ((hand.mask & board_mask) == 0) {
      evaluated_hands.push_back(
          {static_cast<uint16_t>(index),
           static_cast<uint16_t>(evaluate_7cards(
               board[0], board[1], board[2], board[3], board[4], hand.first,
               hand.second))});
    }
  }

  std::sort(evaluated_hands.begin(), evaluated_hands.end(),
            [](const EvaluatedHand &first, const EvaluatedHand &second) {
              return first.rank < second.rank;
            });

  size_t hand_begin = table.hand_indices.size();
  size_t group_begin = table.groups.size();
  for (size_t index = 0; index < evaluated_hands.size(); ++index) {
    table.hand_indices.push_back(evaluated_hands[index].hand_index);
    bool group_ends = index + 1 == evaluated_hands.size() ||
                      evaluated_hands[index].rank !=
                          evaluated_hands[index + 1].rank;
    if (group_ends) {
      table.groups.push_back(
          {evaluated_hands[index].rank, static_cast<uint16_t>(index + 1)});
    }
  }
  table.runout_hand_ranges.push_back(
      {hand_begin, table.hand_indices.size() - hand_begin});
  table.runout_group_ranges.push_back(
      {group_begin, table.groups.size() - group_begin});
}

void TerminalTables::apply_fold(Player evaluated_player,
                                const Range &opponent_range, float payoff,
                                std::vector<float> &values) const {
  assert(opponent_range.hands == hand_tables[static_cast<size_t>(
                                      opponent(evaluated_player))]);
  values.resize(hands(evaluated_player).size());
  apply_fold(evaluated_player, opponent_range.weights, payoff, values);
}

void TerminalTables::apply_fold(Player evaluated_player,
                                std::span<const float> opponent_range,
                                float payoff,
                                std::span<float> values) const {
  auto evaluated_hands = hands(evaluated_player);
  auto opponent_hands = hands(opponent(evaluated_player));
  assert(opponent_range.size() == opponent_hands.size());
  assert(values.size() == evaluated_hands.size());

  float total = 0.0F;
  std::array<float, DECK_SIZE> card_totals{};
  std::array<float, DECK_SIZE * DECK_SIZE> hand_totals{};
  for (size_t index = 0; index < opponent_hands.size(); ++index) {
    const Hand &hand = opponent_hands[index];
    float weight = opponent_range[index];
    total += weight;
    card_totals[hand.first] += weight;
    card_totals[hand.second] += weight;
    hand_totals[hand_key(hand)] += weight;
  }

  for (size_t index = 0; index < evaluated_hands.size(); ++index) {
    const Hand &hand = evaluated_hands[index];
    float compatible_weight =
        total - card_totals[hand.first] - card_totals[hand.second] +
        hand_totals[hand_key(hand)];
    values[index] = payoff * compatible_weight;
  }
}

void TerminalTables::apply_showdown(Player evaluated_player,
                                    size_t runout_index,
                                    const Range &opponent_range,
                                    float win_payoff, float loss_payoff,
                                    std::vector<float> &values) const {
  assert(opponent_range.hands == hand_tables[static_cast<size_t>(
                                      opponent(evaluated_player))]);
  values.resize(hands(evaluated_player).size());
  apply_showdown(evaluated_player, runout_index, opponent_range.weights,
                 win_payoff, loss_payoff, values);
}

void TerminalTables::apply_showdown(Player evaluated_player,
                                    size_t runout_index,
                                    std::span<const float> opponent_range,
                                    float win_payoff, float loss_payoff,
                                    std::span<float> values) const {
  size_t evaluated_index = static_cast<size_t>(evaluated_player);
  size_t opponent_index = static_cast<size_t>(opponent(evaluated_player));
  const RankedHandTable &evaluated_table = ranked[evaluated_index];
  const RankedHandTable &opponent_table = ranked[opponent_index];
  auto evaluated_hands = hands(evaluated_player);
  auto opponent_hands = hands(opponent(evaluated_player));
  assert(opponent_range.size() == opponent_hands.size());
  assert(values.size() == evaluated_hands.size());
  std::fill(values.begin(), values.end(), 0.0F);

  IndexRange evaluated_hand_range =
      evaluated_table.runout_hand_ranges[runout_index];
  IndexRange evaluated_group_range =
      evaluated_table.runout_group_ranges[runout_index];
  IndexRange opponent_hand_range =
      opponent_table.runout_hand_ranges[runout_index];
  IndexRange opponent_group_range =
      opponent_table.runout_group_ranges[runout_index];

  float total = 0.0F;
  std::array<float, DECK_SIZE> card_totals{};
  size_t opponent_group = opponent_group_range.begin + opponent_group_range.count;
  for (size_t evaluated_offset = evaluated_group_range.count;
       evaluated_offset-- > 0;) {
    const RankGroup &group =
        evaluated_table.groups[evaluated_group_range.begin + evaluated_offset];
    while (opponent_group > opponent_group_range.begin &&
           opponent_table.groups[opponent_group - 1].rank > group.rank) {
      --opponent_group;
      uint16_t begin =
          opponent_group == opponent_group_range.begin
              ? 0
              : opponent_table.groups[opponent_group - 1].end;
      uint16_t end = opponent_table.groups[opponent_group].end;
      for (uint16_t offset = begin; offset < end; ++offset) {
        uint16_t hand_index =
            opponent_table.hand_indices[opponent_hand_range.begin + offset];
        const Hand &hand = opponent_hands[hand_index];
        float weight = opponent_range[hand_index];
        total += weight;
        card_totals[hand.first] += weight;
        card_totals[hand.second] += weight;
      }
    }

    uint16_t begin =
        evaluated_offset == 0
            ? 0
            : evaluated_table
                  .groups[evaluated_group_range.begin + evaluated_offset - 1]
                  .end;
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      uint16_t hand_index =
          evaluated_table.hand_indices[evaluated_hand_range.begin + offset];
      const Hand &hand = evaluated_hands[hand_index];
      values[hand_index] =
          win_payoff *
          (total - card_totals[hand.first] - card_totals[hand.second]);
    }
  }

  total = 0.0F;
  card_totals.fill(0.0F);
  opponent_group = opponent_group_range.begin;
  for (size_t evaluated_offset = 0;
       evaluated_offset < evaluated_group_range.count; ++evaluated_offset) {
    const RankGroup &group =
        evaluated_table.groups[evaluated_group_range.begin + evaluated_offset];
    while (opponent_group <
               opponent_group_range.begin + opponent_group_range.count &&
           opponent_table.groups[opponent_group].rank < group.rank) {
      uint16_t begin =
          opponent_group == opponent_group_range.begin
              ? 0
              : opponent_table.groups[opponent_group - 1].end;
      uint16_t end = opponent_table.groups[opponent_group].end;
      for (uint16_t offset = begin; offset < end; ++offset) {
        uint16_t hand_index =
            opponent_table.hand_indices[opponent_hand_range.begin + offset];
        const Hand &hand = opponent_hands[hand_index];
        float weight = opponent_range[hand_index];
        total += weight;
        card_totals[hand.first] += weight;
        card_totals[hand.second] += weight;
      }
      ++opponent_group;
    }

    uint16_t begin =
        evaluated_offset == 0
            ? 0
            : evaluated_table
                  .groups[evaluated_group_range.begin + evaluated_offset - 1]
                  .end;
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      uint16_t hand_index =
          evaluated_table.hand_indices[evaluated_hand_range.begin + offset];
      const Hand &hand = evaluated_hands[hand_index];
      values[hand_index] +=
          loss_payoff *
          (total - card_totals[hand.first] - card_totals[hand.second]);
    }
  }
}

} // namespace spmv_poker
