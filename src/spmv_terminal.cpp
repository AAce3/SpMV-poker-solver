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

size_t lane_major_summary_index(size_t channel, size_t group, size_t lane,
                                size_t group_count, size_t lane_count) {
  return (channel * group_count + group) * lane_count + lane;
}

float lane_major_prefix_at(std::span<const float> prefix, size_t channel,
                           size_t group, size_t lane, size_t group_count,
                           size_t lane_count) {
  return prefix[lane_major_summary_index(channel, group, lane, group_count,
                                         lane_count)];
}

} // namespace

void RiverTerminalOperator::mask_board_reaches(BoardIndex, Player,
                                               std::span<const float> input,
                                               std::span<float> output) const {
  assert(input.size() == output.size());
  std::ranges::copy(input, output.begin());
}

void RiverTerminalOperator::mask_board_values(BoardIndex, Player,
                                              std::span<float>) const {}

std::span<const Hand> RankSummaryTerminalOperator::hands(
    Player player) const {
  return tables_.hands(player);
}

RankSummaryTerminalOperator::RankSummaryTerminalOperator(
    const TerminalTables &tables, uint64_t flop_mask)
    : RankSummaryTerminalOperator(tables, flop_mask, FLOP_CARD_COUNT) {}

RankSummaryTerminalOperator::RankSummaryTerminalOperator(
    const TerminalTables &tables, uint64_t public_mask,
    size_t public_card_count)
    : tables_(tables) {
  RunoutIndex boards(public_mask, public_card_count);
  std::unordered_map<uint64_t, uint32_t> runout_by_mask;
  runout_by_mask.reserve(tables.runouts.size());
  for (size_t runout = 0; runout < tables.runouts.size(); ++runout) {
    runout_by_mask.emplace(make_mask(tables.runouts[runout]),
                           static_cast<uint32_t>(runout));
  }

  std::vector<BoardMetadata> metadata_by_runout(tables.runouts.size());
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
    metadata_by_runout[runout] = metadata;
    max_rank_group_count_ =
        std::max(max_rank_group_count_,
                 static_cast<size_t>(metadata.rank_group_count));
  }

  terminal_plans_.resize(boards.board_count(Street::River));
  for (BoardIndex board = 0; board < boards.board_count(Street::River);
       ++board) {
    auto found = runout_by_mask.find(boards.board_mask(Street::River, board));
    if (found == runout_by_mask.end()) {
      throw std::invalid_argument(
          "terminal tables do not contain every river board for the flop");
    }
    terminal_plans_[board].runout_index = found->second;
    terminal_plans_[board].metadata = metadata_by_runout[found->second];
  }

  for (Player evaluated_player : {Player::Hero, Player::Villain}) {
    size_t player_index = static_cast<size_t>(evaluated_player);
    std::array<int32_t, DECK_SIZE * DECK_SIZE> opponent_by_hand;
    opponent_by_hand.fill(-1);
    auto hands = tables_.hands(evaluated_player);
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

const RankSummaryTerminalOperator::CompiledTerminalPlan &
RankSummaryTerminalOperator::terminal_plan(BoardIndex board) const {
  assert(board < terminal_plans_.size());
  return terminal_plans_[board];
}

size_t RankSummaryTerminalOperator::showdown_summary_scratch_floats(
    std::span<const BoardIndex> boards, size_t showdown_count) const {
  size_t max_group_count = 0;
  for (BoardIndex board : boards) {
    const CompiledTerminalPlan &plan = terminal_plan(board);
    max_group_count = std::max(
        max_group_count,
        static_cast<size_t>(plan.metadata.rank_group_count));
  }
  return boards.size() * std::min(showdown_count, TERMINAL_TILE_SIZE) *
         SUMMARY_CHANNEL_COUNT * max_group_count;
}

void RankSummaryTerminalOperator::reserve_showdown_summary_scratch(
    std::span<const BoardIndex> boards, size_t showdown_count) const {
  showdown_summary_scratch_.resize(
      showdown_summary_scratch_floats(boards, showdown_count));
}

void RankSummaryTerminalOperator::evaluate_folds(
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

void RankSummaryTerminalOperator::mask_board_reaches(
    BoardIndex board, Player player, std::span<const float> input,
    std::span<float> output) const {
  assert(input.size() == output.size());
  const CompiledTerminalPlan &plan = terminal_plan(board);
  auto hands = tables_.hands(player);
  assert(input.size() >= hands.size());
  std::ranges::copy(input, output.begin());
  uint64_t board_mask = make_mask(tables_.runouts[plan.runout_index]);
  for (size_t hand = 0; hand < hands.size(); ++hand) {
    output[hand] *= (hands[hand].mask & board_mask) == 0 ? 1.0F : 0.0F;
  }
  std::ranges::fill(output.subspan(hands.size()), 0.0F);
}

void RankSummaryTerminalOperator::mask_board_values(
    BoardIndex board, Player player, std::span<float> values) const {
  const CompiledTerminalPlan &plan = terminal_plan(board);
  auto hands = tables_.hands(player);
  assert(values.size() >= hands.size());
  uint64_t board_mask = make_mask(tables_.runouts[plan.runout_index]);
  for (size_t hand = 0; hand < hands.size(); ++hand) {
    values[hand] *= (hands[hand].mask & board_mask) == 0 ? 1.0F : 0.0F;
  }
  std::ranges::fill(values.subspan(hands.size()), 0.0F);
}

void RankSummaryTerminalOperator::evaluate_showdowns_lane_major(
    BoardIndex board, Player evaluated_player,
    std::span<const float> lane_major_opponent_reaches, size_t terminal_count,
    std::span<const CompiledShowdown> showdowns,
    std::span<float> lane_major_values) const {
  const CompiledTerminalPlan &plan = terminal_plan(board);
  const BoardMetadata &metadata = plan.metadata;
  size_t group_count = metadata.rank_group_count;
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  size_t evaluated_index = static_cast<size_t>(evaluated_player);
  size_t opponent_index = static_cast<size_t>(opponent(evaluated_player));
  const RankedHandTable &evaluated_ranked = tables_.ranked[evaluated_index];
  const RankedHandTable &opponent_ranked = tables_.ranked[opponent_index];
  IndexRange evaluated_hand_range =
      evaluated_ranked.runout_hand_ranges[plan.runout_index];
  IndexRange evaluated_group_range =
      evaluated_ranked.runout_group_ranges[plan.runout_index];
  IndexRange opponent_hand_range =
      opponent_ranked.runout_hand_ranges[plan.runout_index];
  IndexRange opponent_group_range =
      opponent_ranked.runout_group_ranges[plan.runout_index];
  assert(showdowns.size() == terminal_count);
  assert(lane_major_opponent_reaches.size() ==
         showdowns.size() * opponent_hands.size());
  assert(lane_major_values.size() == showdowns.size() * evaluated_hands.size());
  std::ranges::fill(lane_major_values, 0.0F);
  if (terminal_count == 0 || group_count == 0) {
    return;
  }

  Tile total{};
  std::array<Tile, DECK_SIZE> card_totals{};

  size_t opponent_group = opponent_group_range.begin + opponent_group_range.count;
  for (size_t evaluated_offset = evaluated_group_range.count; evaluated_offset-- > 0;) {
    const RankGroup &group =
        evaluated_ranked.groups[evaluated_group_range.begin + evaluated_offset];
    while (opponent_group > opponent_group_range.begin &&
           opponent_ranked.groups[opponent_group - 1].rank > group.rank) {
      --opponent_group;
      uint16_t begin =
          opponent_group == opponent_group_range.begin
              ? 0
              : opponent_ranked.groups[opponent_group - 1].end;
      uint16_t end = opponent_ranked.groups[opponent_group].end;
      for (uint16_t offset = begin; offset < end; ++offset) {
        uint16_t hand_index =
            opponent_ranked.hand_indices[opponent_hand_range.begin + offset];
        const Hand &hand = opponent_hands[hand_index];
        const float *hand_reaches =
            lane_major_opponent_reaches.data() + hand_index * terminal_count;
        for (size_t lane = 0; lane < terminal_count; ++lane) {
          float reach = hand_reaches[lane];
          total[lane] += reach;
          card_totals[hand.first][lane] += reach;
          card_totals[hand.second][lane] += reach;
        }
      }
    }

    uint16_t begin =
        evaluated_offset == 0
            ? 0
            : evaluated_ranked
                  .groups[evaluated_group_range.begin + evaluated_offset - 1]
                  .end;
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      uint16_t hand_index =
          evaluated_ranked.hand_indices[evaluated_hand_range.begin + offset];
      const Hand &hand = evaluated_hands[hand_index];
      float *values = lane_major_values.data() + hand_index * terminal_count;
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float compatible =
            total[lane] - card_totals[hand.first][lane] -
            card_totals[hand.second][lane];
        values[lane] = showdowns[lane].win_payoff * compatible;
      }
    }
  }

  total.fill(0.0F);
  for (auto &card_total : card_totals) {
    card_total.fill(0.0F);
  }
  opponent_group = opponent_group_range.begin;
  for (size_t evaluated_offset = 0;
       evaluated_offset < evaluated_group_range.count; ++evaluated_offset) {
    const RankGroup &group =
        evaluated_ranked.groups[evaluated_group_range.begin + evaluated_offset];
    while (opponent_group <
               opponent_group_range.begin + opponent_group_range.count &&
           opponent_ranked.groups[opponent_group].rank < group.rank) {
      uint16_t begin =
          opponent_group == opponent_group_range.begin
              ? 0
              : opponent_ranked.groups[opponent_group - 1].end;
      uint16_t end = opponent_ranked.groups[opponent_group].end;
      for (uint16_t offset = begin; offset < end; ++offset) {
        uint16_t hand_index =
            opponent_ranked.hand_indices[opponent_hand_range.begin + offset];
        const Hand &hand = opponent_hands[hand_index];
        const float *hand_reaches =
            lane_major_opponent_reaches.data() + hand_index * terminal_count;
        for (size_t lane = 0; lane < terminal_count; ++lane) {
          float reach = hand_reaches[lane];
          total[lane] += reach;
          card_totals[hand.first][lane] += reach;
          card_totals[hand.second][lane] += reach;
        }
      }
      ++opponent_group;
    }

    uint16_t begin =
        evaluated_offset == 0
            ? 0
            : evaluated_ranked
                  .groups[evaluated_group_range.begin + evaluated_offset - 1]
                  .end;
    for (uint16_t offset = begin; offset < group.end; ++offset) {
      uint16_t hand_index =
          evaluated_ranked.hand_indices[evaluated_hand_range.begin + offset];
      const Hand &hand = evaluated_hands[hand_index];
      float *values = lane_major_values.data() + hand_index * terminal_count;
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        float compatible =
            total[lane] - card_totals[hand.first][lane] -
            card_totals[hand.second][lane];
        values[lane] += showdowns[lane].loss_payoff * compatible;
      }
    }
  }
}

void RankSummaryTerminalOperator::evaluate_fold_board_batch(
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

void RankSummaryTerminalOperator::evaluate_showdowns(
    BoardIndex board, Player evaluated_player,
    std::span<const float> opponent_reaches, size_t opponent_hand_stride,
    std::span<const CompiledShowdown> showdowns, std::span<float> values,
    size_t evaluated_hand_stride) const {
  auto evaluated_hands = tables_.hands(evaluated_player);
  auto opponent_hands = tables_.hands(opponent(evaluated_player));
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  assert(opponent_reaches.size() == showdowns.size() * opponent_hand_stride);
  assert(values.size() == showdowns.size() * evaluated_hand_stride);
  std::ranges::fill(values, 0.0F);

  if (showdowns.empty()) {
    return;
  }

  size_t max_terminal_count =
      std::min(TERMINAL_TILE_SIZE, showdowns.size());
  showdown_lane_major_reach_scratch_.resize(opponent_hands.size() *
                                            max_terminal_count);
  showdown_lane_major_value_scratch_.resize(evaluated_hands.size() *
                                            max_terminal_count);

  for (size_t terminal_begin = 0; terminal_begin < showdowns.size();
       terminal_begin += TERMINAL_TILE_SIZE) {
    size_t terminal_count =
        std::min(TERMINAL_TILE_SIZE, showdowns.size() - terminal_begin);
    std::span<float> lane_major_reaches(
        showdown_lane_major_reach_scratch_.data(),
        opponent_hands.size() * terminal_count);
    std::span<float> lane_major_values(
        showdown_lane_major_value_scratch_.data(),
        evaluated_hands.size() * terminal_count);

    for (size_t hand_index = 0; hand_index < opponent_hands.size();
         ++hand_index) {
      float *lane_reaches =
          lane_major_reaches.data() + hand_index * terminal_count;
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        lane_reaches[lane] = opponent_reaches[(terminal_begin + lane) *
                                                  opponent_hand_stride +
                                              hand_index];
      }
    }

    evaluate_showdowns_lane_major(
        board, evaluated_player, lane_major_reaches, terminal_count,
        showdowns.subspan(terminal_begin, terminal_count), lane_major_values);

    for (size_t hand_index = 0; hand_index < evaluated_hands.size();
         ++hand_index) {
      const float *lane_values =
          lane_major_values.data() + hand_index * terminal_count;
      for (size_t lane = 0; lane < terminal_count; ++lane) {
        values[(terminal_begin + lane) * evaluated_hand_stride + hand_index] =
            lane_values[lane];
      }
    }
  }
}

void RankSummaryTerminalOperator::evaluate_showdown_board_batch(
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
  assert(opponent_hand_stride >= opponent_hands.size());
  assert(evaluated_hand_stride >= evaluated_hands.size());
  std::ranges::fill(values, 0.0F);

  if (board_count == 0 || showdowns.empty()) {
    return;
  }

  std::vector<float> board_reaches(showdowns.size() * opponent_hand_stride);
  std::vector<float> board_values(showdowns.size() * evaluated_hand_stride);
  for (BoardIndex board = 0; board < boards.size(); ++board) {
    for (size_t terminal = 0; terminal < showdowns.size(); ++terminal) {
      auto source = opponent_reaches.subspan(
          (terminal * board_count + board) * opponent_hand_stride,
          opponent_hand_stride);
      std::ranges::copy(
          source, board_reaches.begin() + terminal * opponent_hand_stride);
    }
    evaluate_showdowns(boards[board], evaluated_player, board_reaches,
                       opponent_hand_stride, showdowns, board_values,
                       evaluated_hand_stride);
    for (size_t terminal = 0; terminal < showdowns.size(); ++terminal) {
      auto source = std::span<const float>(board_values)
                        .subspan(terminal * evaluated_hand_stride,
                                 evaluated_hand_stride);
      std::ranges::copy(source, values.begin() +
                                  (terminal * board_count + board) *
                                      evaluated_hand_stride);
    }
  }
}

} // namespace spmv_poker
