#include "spmv_poker/spmv_terminal.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <unordered_map>

namespace spmv_poker {

namespace {

constexpr size_t GLOBAL_CHANNEL = DECK_SIZE;
constexpr size_t SUMMARY_CHANNEL_COUNT = DECK_SIZE + 1;

using Tile = std::array<float, TERMINAL_TILE_SIZE>;
using CardTiles = std::array<Tile, DECK_SIZE>;

constexpr size_t hand_key(const Hand &hand) {
  return static_cast<size_t>(hand.first) * DECK_SIZE + hand.second;
}

std::vector<uint16_t> hand_ranks(const TerminalTables &tables, Player player,
                                 size_t runout) {
  const RankedHandTable &ranked = tables.ranked[static_cast<size_t>(player)];
  IndexRange hand_range = ranked.runout_hand_ranges[runout];
  IndexRange group_range = ranked.runout_group_ranges[runout];
  std::vector<uint16_t> result(tables.hands(player).size());
  uint16_t begin = 0;
  for (size_t group_offset = 0; group_offset < group_range.count;
       ++group_offset) {
    const RankGroup &group = ranked.groups[group_range.begin + group_offset];
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      result[ranked.hand_indices[hand_range.begin + offset]] = group.rank;
    }
    begin = group.end;
  }
  return result;
}

size_t summary_index(size_t channel, size_t group, size_t lane,
                     size_t group_count) {
  return (channel * group_count + group) * TERMINAL_TILE_SIZE + lane;
}

float prefix_at(std::span<const float> prefix, size_t channel, size_t group,
                size_t lane, size_t group_count) {
  return prefix[summary_index(channel, group, lane, group_count)];
}

} // namespace

TiledRankSummaryRiverTerminalOperator::TiledRankSummaryRiverTerminalOperator(
    const TerminalTables &tables, uint64_t flop_mask)
    : tables_(tables) {
  RunoutIndex boards(flop_mask);
  std::unordered_map<uint64_t, uint32_t> runout_by_mask;
  runout_by_mask.reserve(tables.runouts.size());
  for (size_t runout = 0; runout < tables.runouts.size(); ++runout) {
    runout_by_mask.emplace(make_mask(tables.runouts[runout]),
                           static_cast<uint32_t>(runout));
  }

  runout_by_river_board_.reserve(boards.board_count(Street::River));
  for (BoardIndex board = 0; board < boards.board_count(Street::River);
       ++board) {
    auto found = runout_by_mask.find(boards.board_mask(Street::River, board));
    if (found == runout_by_mask.end()) {
      throw std::invalid_argument(
          "terminal tables do not contain every river board for the flop");
    }
    runout_by_river_board_.push_back(found->second);
  }

  board_metadata_.reserve(tables.runouts.size());
  for (size_t runout = 0; runout < tables.runouts.size(); ++runout) {
    std::array<std::vector<uint16_t>, 2> ranks{
        hand_ranks(tables, Player::Hero, runout),
        hand_ranks(tables, Player::Villain, runout),
    };
    std::vector<uint16_t> merged_ranks;
    for (const auto &player_ranks : ranks) {
      for (uint16_t rank : player_ranks) {
        if (rank != 0) {
          merged_ranks.push_back(rank);
        }
      }
    }
    std::sort(merged_ranks.begin(), merged_ranks.end(), std::greater<>());
    merged_ranks.erase(std::unique(merged_ranks.begin(), merged_ranks.end()),
                       merged_ranks.end());
    assert(merged_ranks.size() < INVALID_RANK_GROUP);

    BoardMetadata metadata;
    metadata.rank_group_count = static_cast<uint16_t>(merged_ranks.size());
    for (Player player : {Player::Hero, Player::Villain}) {
      size_t player_index = static_cast<size_t>(player);
      auto &all_groups = rank_groups_by_board_hand_[player_index];
      size_t group_begin = all_groups.size();
      all_groups.resize(group_begin + ranks[player_index].size(),
                        INVALID_RANK_GROUP);
      for (size_t hand = 0; hand < ranks[player_index].size(); ++hand) {
        uint16_t rank = ranks[static_cast<size_t>(player)][hand];
        if (rank != 0) {
          all_groups[group_begin + hand] = static_cast<uint16_t>(
              std::lower_bound(merged_ranks.begin(), merged_ranks.end(), rank,
                               std::greater<>()) -
              merged_ranks.begin());
        }
      }
      metadata.player_rank_groups[player_index] =
          IndexRange{group_begin, ranks[player_index].size()};
    }
    board_metadata_.push_back(std::move(metadata));
    max_rank_group_count_ =
        std::max(max_rank_group_count_,
                 static_cast<size_t>(metadata.rank_group_count));
  }

  for (Player evaluated_player : {Player::Hero, Player::Villain}) {
    std::array<int32_t, DECK_SIZE * DECK_SIZE> opponent_by_hand;
    opponent_by_hand.fill(-1);
    auto opponent_hands = tables_.hands(opponent(evaluated_player));
    for (size_t hand = 0; hand < opponent_hands.size(); ++hand) {
      opponent_by_hand[hand_key(opponent_hands[hand])] =
          static_cast<int32_t>(hand);
    }
    auto evaluated_hands = tables_.hands(evaluated_player);
    auto &matches =
        matching_opponent_hands_[static_cast<size_t>(evaluated_player)];
    matches.reserve(evaluated_hands.size());
    for (const Hand &hand : evaluated_hands) {
      matches.push_back(opponent_by_hand[hand_key(hand)]);
    }
  }
}

size_t TiledRankSummaryRiverTerminalOperator::runout_for_board(
    BoardIndex board) const {
  assert(board < runout_by_river_board_.size());
  return runout_by_river_board_[board];
}

size_t TiledRankSummaryRiverTerminalOperator::showdown_summary_scratch_floats(
    std::span<const BoardIndex> boards, size_t showdown_count) const {
  size_t max_group_count = 0;
  for (BoardIndex board : boards) {
    size_t runout = runout_for_board(board);
    max_group_count = std::max(
        max_group_count,
        static_cast<size_t>(board_metadata_[runout].rank_group_count));
  }
  return boards.size() * std::min(showdown_count, TERMINAL_TILE_SIZE) *
         SUMMARY_CHANNEL_COUNT * max_group_count;
}

void TiledRankSummaryRiverTerminalOperator::reserve_showdown_summary_scratch(
    std::span<const BoardIndex> boards, size_t showdown_count) const {
  showdown_summary_scratch_.resize(
      showdown_summary_scratch_floats(boards, showdown_count));
}

void TiledRankSummaryRiverTerminalOperator::evaluate_folds(
    BoardIndex, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const float> payoffs, std::span<float> values,
    size_t evaluated_hand_stride) const {
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  assert(opponent_reaches.size() == payoffs.size() * opponent_hand_stride);
  assert(values.size() == payoffs.size() * evaluated_hand_stride);
  std::ranges::fill(values, 0.0F);

  for (size_t terminal_begin = 0; terminal_begin < payoffs.size();
       terminal_begin += TERMINAL_TILE_SIZE) {
    size_t terminal_count =
        std::min(TERMINAL_TILE_SIZE, payoffs.size() - terminal_begin);
    Tile total{};
    CardTiles card_totals{};

    for (size_t opponent_hand = 0; opponent_hand < opponent_hands.size();
         ++opponent_hand) {
      const Hand &hand = opponent_hands[opponent_hand];
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float reach = opponent_reaches[(terminal_begin + lane) *
                                           opponent_hand_stride +
                                       opponent_hand];
        total[lane] += reach;
        card_totals[hand.first][lane] += reach;
        card_totals[hand.second][lane] += reach;
      }
    }

    for (size_t evaluated_hand = 0; evaluated_hand < evaluated_hands.size();
         ++evaluated_hand) {
      const Hand &hand = evaluated_hands[evaluated_hand];
      int32_t matching_opponent =
          matching_opponent_hands_[static_cast<size_t>(evaluated_player)]
                                  [evaluated_hand];
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float identical =
            matching_opponent < 0
                ? 0.0F
                : opponent_reaches[(terminal_begin + lane) *
                                       opponent_hand_stride +
                                   static_cast<size_t>(matching_opponent)];
        float compatible =
            total[lane] - card_totals[hand.first][lane] -
            card_totals[hand.second][lane] + identical;
        values[(terminal_begin + lane) * evaluated_hand_stride +
               evaluated_hand] = payoffs[terminal_begin + lane] * compatible;
      }
    }
  }
}

void TiledRankSummaryRiverTerminalOperator::evaluate_fold_board_batch(
    std::span<const BoardIndex> boards, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const float> payoffs, std::span<float> values,
    size_t evaluated_hand_stride) const {
  size_t board_count = boards.size();
  assert(opponent_reaches.size() ==
         payoffs.size() * board_count * opponent_hand_stride);
  assert(values.size() ==
         payoffs.size() * board_count * evaluated_hand_stride);
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  std::ranges::fill(values, 0.0F);

  size_t column_count = board_count * payoffs.size();
  for (size_t column = 0; column < column_count; ++column) {
    size_t terminal = column / board_count;
    std::span<const float> column_reaches =
        opponent_reaches.subspan(column * opponent_hand_stride,
                                 opponent_hand_stride);

    float total = 0.0F;
    std::array<float, DECK_SIZE> card_totals{};
    for (size_t opponent_hand = 0; opponent_hand < opponent_hands.size();
         ++opponent_hand) {
      const Hand &hand = opponent_hands[opponent_hand];
      float reach = column_reaches[opponent_hand];
      total += reach;
      card_totals[hand.first] += reach;
      card_totals[hand.second] += reach;
    }

    float *column_values = values.data() + column * evaluated_hand_stride;
    for (size_t evaluated_hand = 0; evaluated_hand < evaluated_hands.size();
         ++evaluated_hand) {
      const Hand &hand = evaluated_hands[evaluated_hand];
      int32_t matching_opponent =
          matching_opponent_hands_[static_cast<size_t>(evaluated_player)]
                                  [evaluated_hand];
      float identical =
          matching_opponent < 0 ? 0.0F : column_reaches[static_cast<size_t>(
                                       matching_opponent)];
      float compatible = total - card_totals[hand.first] -
                         card_totals[hand.second] + identical;
      column_values[evaluated_hand] = payoffs[terminal] * compatible;
    }
  }
}

void TiledRankSummaryRiverTerminalOperator::evaluate_showdowns(
    BoardIndex board, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const CompiledShowdown> showdowns, std::span<float> values,
    size_t evaluated_hand_stride) const {
  size_t runout = runout_for_board(board);
  const BoardMetadata &metadata = board_metadata_[runout];
  size_t group_count = metadata.rank_group_count;
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  size_t evaluated_index = static_cast<size_t>(evaluated_player);
  size_t opponent_index = static_cast<size_t>(opponent(evaluated_player));
  auto evaluated_groups = metadata.player_rank_groups[evaluated_index].view(
      rank_groups_by_board_hand_[evaluated_index]);
  auto opponent_groups = metadata.player_rank_groups[opponent_index].view(
      rank_groups_by_board_hand_[opponent_index]);
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  assert(opponent_reaches.size() == showdowns.size() * opponent_hand_stride);
  assert(values.size() == showdowns.size() * evaluated_hand_stride);
  std::ranges::fill(values, 0.0F);

  for (size_t terminal_begin = 0; terminal_begin < showdowns.size();
       terminal_begin += TERMINAL_TILE_SIZE) {
    size_t terminal_count =
        std::min(TERMINAL_TILE_SIZE, showdowns.size() - terminal_begin);
    std::vector<float> prefix(SUMMARY_CHANNEL_COUNT * group_count *
                                  TERMINAL_TILE_SIZE,
                              0.0F);

    // Stage 1: scatter opponent reaches into global and blocker rank summaries.
    for (size_t hand_index = 0; hand_index < opponent_hands.size();
         ++hand_index) {
      uint16_t group = opponent_groups[hand_index];
      if (group == INVALID_RANK_GROUP) {
        continue;
      }
      const Hand &hand = opponent_hands[hand_index];
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float reach = opponent_reaches[(terminal_begin + lane) *
                                           opponent_hand_stride +
                                       hand_index];
        prefix[summary_index(GLOBAL_CHANNEL, group, lane, group_count)] +=
            reach;
        prefix[summary_index(hand.first, group, lane, group_count)] += reach;
        prefix[summary_index(hand.second, group, lane, group_count)] += reach;
      }
    }

    // Stage 2: independently prefix-scan every global/card summary channel.
    for (size_t channel = 0; channel < SUMMARY_CHANNEL_COUNT; ++channel) {
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float running = 0.0F;
        for (size_t group = 0; group < group_count; ++group) {
          size_t index = summary_index(channel, group, lane, group_count);
          running += prefix[index];
          prefix[index] = running;
        }
      }
    }

    // Stage 3: independently produce each evaluated-hand value.
    for (size_t hand_index = 0; hand_index < evaluated_hands.size();
         ++hand_index) {
      uint16_t group = evaluated_groups[hand_index];
      if (group == INVALID_RANK_GROUP) {
        continue;
      }
      const Hand &hand = evaluated_hands[hand_index];
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float global_total =
            prefix_at(prefix, GLOBAL_CHANNEL, group_count - 1, lane,
                      group_count);
        float global_through =
            prefix_at(prefix, GLOBAL_CHANNEL, group, lane, group_count);
        float global_weaker =
            group == 0 ? 0.0F
                       : prefix_at(prefix, GLOBAL_CHANNEL, group - 1, lane,
                                   group_count);

        auto blocked = [&](size_t card, size_t blocked_group) {
          return prefix_at(prefix, card, blocked_group, lane, group_count);
        };
        float blocked_weaker =
            group == 0 ? 0.0F
                       : blocked(hand.first, group - 1) +
                             blocked(hand.second, group - 1);
        float blocked_stronger =
            (prefix_at(prefix, hand.first, group_count - 1, lane, group_count) -
             blocked(hand.first, group)) +
            (prefix_at(prefix, hand.second, group_count - 1, lane,
                       group_count) -
             blocked(hand.second, group));

        float weaker = global_weaker - blocked_weaker;
        float stronger = global_total - global_through - blocked_stronger;
        const CompiledShowdown &showdown = showdowns[terminal_begin + lane];
        values[(terminal_begin + lane) * evaluated_hand_stride + hand_index] =
            showdown.win_payoff * weaker + showdown.loss_payoff * stronger;
      }
    }
  }
}

void TiledRankSummaryRiverTerminalOperator::evaluate_showdown_board_batch(
    std::span<const BoardIndex> boards, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const CompiledShowdown> showdowns, std::span<float> values,
    size_t evaluated_hand_stride) const {
  size_t board_count = boards.size();
  assert(opponent_reaches.size() ==
         showdowns.size() * board_count * opponent_hand_stride);
  assert(values.size() ==
         showdowns.size() * board_count * evaluated_hand_stride);
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  size_t evaluated_index = static_cast<size_t>(evaluated_player);
  size_t opponent_index = static_cast<size_t>(opponent(evaluated_player));
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  std::ranges::fill(values, 0.0F);

  size_t max_group_count = 0;
  for (BoardIndex board : boards) {
    size_t runout = runout_for_board(board);
    max_group_count = std::max(
        max_group_count,
        static_cast<size_t>(board_metadata_[runout].rank_group_count));
  }
  if (board_count == 0 || showdowns.empty() || max_group_count == 0) {
    return;
  }

  size_t required = showdown_summary_scratch_floats(boards, showdowns.size());
  if (showdown_summary_scratch_.size() < required) {
    showdown_summary_scratch_.resize(required);
  }
  std::fill_n(showdown_summary_scratch_.begin(), required, 0.0F);
  std::span<float> scratch(showdown_summary_scratch_.data(), required);

  for (size_t terminal_begin = 0; terminal_begin < showdowns.size();
       terminal_begin += TERMINAL_TILE_SIZE) {
    size_t terminal_count =
        std::min(TERMINAL_TILE_SIZE, showdowns.size() - terminal_begin);
    size_t tile_columns = board_count * terminal_count;
    size_t tile_floats = tile_columns * SUMMARY_CHANNEL_COUNT *
                         max_group_count;
    std::fill_n(scratch.begin(), tile_floats, 0.0F);
    std::span<float> tile_scratch(scratch.data(), tile_floats);

    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      for (size_t board = 0; board < board_count; ++board) {
        size_t column = terminal * board_count + board;
        size_t global_column = (terminal_begin + terminal) * board_count + board;
        size_t runout = runout_for_board(boards[board]);
        const BoardMetadata &metadata = board_metadata_[runout];
        size_t group_count = metadata.rank_group_count;
        if (group_count == 0) {
          continue;
        }
        auto evaluated_groups =
            metadata.player_rank_groups[evaluated_index]
                .view(rank_groups_by_board_hand_[evaluated_index]);
        auto opponent_groups =
            metadata.player_rank_groups[opponent_index]
                .view(rank_groups_by_board_hand_[opponent_index]);

        float *column_base =
            tile_scratch.data() + column * SUMMARY_CHANNEL_COUNT *
                max_group_count;
        std::span<const float> column_reaches =
            opponent_reaches.subspan(global_column * opponent_hand_stride,
                                     opponent_hand_stride);

        for (size_t opponent_hand = 0; opponent_hand < opponent_hands.size();
             ++opponent_hand) {
          uint16_t group = opponent_groups[opponent_hand];
          if (group == INVALID_RANK_GROUP) {
            continue;
          }
          const Hand &hand = opponent_hands[opponent_hand];
          float reach = column_reaches[opponent_hand];
          column_base[GLOBAL_CHANNEL * max_group_count + group] += reach;
          column_base[hand.first * max_group_count + group] += reach;
          column_base[hand.second * max_group_count + group] += reach;
        }

        for (size_t channel = 0; channel < SUMMARY_CHANNEL_COUNT; ++channel) {
          float *channel_base = column_base + channel * max_group_count;
          float running = 0.0F;
          for (size_t group = 0; group < group_count; ++group) {
            running += channel_base[group];
            channel_base[group] = running;
          }
        }
      }
    }

    for (size_t terminal = 0; terminal < terminal_count; ++terminal) {
      const CompiledShowdown &showdown = showdowns[terminal_begin + terminal];
      for (size_t board = 0; board < board_count; ++board) {
        size_t column = terminal * board_count + board;
        size_t runout = runout_for_board(boards[board]);
        const BoardMetadata &metadata = board_metadata_[runout];
        size_t group_count = metadata.rank_group_count;
        if (group_count == 0) {
          continue;
        }
        auto evaluated_groups =
            metadata.player_rank_groups[evaluated_index]
                .view(rank_groups_by_board_hand_[evaluated_index]);

        float *column_base =
            tile_scratch.data() + column * SUMMARY_CHANNEL_COUNT *
                max_group_count;
        float *column_values =
            values.data() + (terminal_begin + terminal) * board_count *
                                 evaluated_hand_stride +
            board * evaluated_hand_stride;
        float global_total =
            column_base[GLOBAL_CHANNEL * max_group_count + group_count - 1];

        for (size_t evaluated_hand = 0; evaluated_hand < evaluated_hands.size();
             ++evaluated_hand) {
          uint16_t group = evaluated_groups[evaluated_hand];
          if (group == INVALID_RANK_GROUP) {
            continue;
          }
          const Hand &hand = evaluated_hands[evaluated_hand];
          float global_through =
              column_base[GLOBAL_CHANNEL * max_group_count + group];
          float global_weaker =
              group == 0 ? 0.0F
                         : column_base[GLOBAL_CHANNEL * max_group_count +
                                       group - 1];

          auto blocked = [&](size_t card, size_t blocked_group) {
            return column_base[card * max_group_count + blocked_group];
          };
          float blocked_weaker =
              group == 0 ? 0.0F
                         : blocked(hand.first, group - 1) +
                               blocked(hand.second, group - 1);
          float blocked_stronger =
              (blocked(hand.first, group_count - 1) - blocked(hand.first, group)) +
              (blocked(hand.second, group_count - 1) -
               blocked(hand.second, group));

          float weaker = global_weaker - blocked_weaker;
          float stronger = global_total - global_through - blocked_stronger;
          column_values[evaluated_hand] =
              showdown.win_payoff * weaker + showdown.loss_payoff * stronger;
        }
      }
    }
  }
}

} // namespace spmv_poker
