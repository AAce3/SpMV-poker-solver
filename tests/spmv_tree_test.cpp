#include "spmv_poker/tree_compiler.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace spmv_poker;

void check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

StreetTopology make_topology() {
  StreetTopology topology(Street::River, 2, 3);
  NodeIndex showdown = topology.add_showdown_node(2.0F, -2.0F);
  NodeIndex boundary = topology.add_boundary_node();

  std::array villain_children{showdown};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain, showdown, boundary};
  topology.root = topology.add_player_node(Player::Hero, hero_children);
  return topology;
}

void test_shared_topology_and_child_order() {
  StreetTopology topology = make_topology();

  check(topology.nodes.size() == 4, "implicit folds allocate no nodes");
  check(topology.fold_payoffs.size() == 1,
        "fold payoff array sizes compact fold buffers");
  check(topology.showdown_count == 1, "showdowns occupy the first node group");
  check(topology.boundary_count == 1, "boundaries follow showdowns");
  check(topology.player_begin() == 2, "players occupy the final node group");
  check(topology.showdowns().size() == 1,
        "showdown view derives from grouped counts");
  check(topology.boundaries().size() == 1,
        "boundary view derives from grouped counts");
  check(topology.players().size() == 2,
        "player view consumes the remaining node suffix");
  check(topology.player_reach_counts[0] == 1,
        "hero player nodes receive compact hero reach slots");
  check(topology.player_reach_counts[1] == 1,
        "villain player nodes receive compact villain reach slots");
  check(topology.children.size() == 4,
        "only continuing actions allocate child entries");
  check(topology.child_endpoints.size() == topology.children.size(),
        "continuing actions store parallel endpoint metadata");
  check(topology.endpoint_nodes.size() == topology.endpoint_count,
        "each endpoint identifies its terminal or boundary node");

  const StreetNode &root = topology.nodes[topology.root];
  auto root_children = topology.child_nodes(root);
  check(root.type == StreetNodeType::Player, "root is a player node");
  check(root_children.size() == 3, "root stores every action");
  check(root_children[0] == topology.player_begin(),
        "action zero preserves remapped player child");
  check(topology.is_showdown(root_children[1]),
        "action one preserves showdown child category");
  check(topology.is_boundary(root_children[2]),
        "action two preserves boundary child category");

  const StreetNode &villain = topology.nodes[topology.player_begin()];
  check(villain.cfr_offset_within_board == 0,
        "CFR offset identifies node state within a board");
  check(villain.action_count() == 2,
        "implicit fold remains part of action state");
  check(villain.is_fold_action(0), "fold uses action zero");
  check(villain.fold_index == 0, "fold receives a compact output slot");
  check(villain.reach_index == 0,
        "player node receives a compact acting-player reach slot");
  check(topology.fold_payoff(villain) == -1.0F,
        "fold slot indexes its terminal payoff");
  check(topology.child_nodes(villain).size() == 1,
        "fold allocates no child node");
  check(topology.is_showdown(topology.child_for_action(villain, 1)),
        "continuing action maps to child node");
  check(topology.endpoint_for_action(villain, 1) != INVALID_INDEX,
        "terminal actions receive path-specific endpoints");

  check(topology.nodes[0].type == StreetNodeType::Showdown,
        "showdown records are physically first");
  check(topology.nodes[1].type == StreetNodeType::StreetBoundary,
        "boundary records follow showdowns");
  check(!root.has_fold(), "missing fold slot represents no fold action");
}

void test_board_node_action_padded_hand_layout() {
  StreetTree game(make_topology(), make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  BoardIndex first = 0;
  BoardIndex second = 1;

  check(game.compiled.nodes.size() == 2,
        "street tree stores only compiled player nodes");
  check(game.compiled.state_entries_per_board == 160,
        "board state pads each player node action to 32 hands");
  check(game.regrets.size() == RIVER_BOARD_COUNT * 160,
        "every derived river board receives separate regrets");
  check(game.cumulative_strategy.size() == RIVER_BOARD_COUNT * 160,
        "every derived river board receives separate strategy sums");

  uint32_t villain = 0;
  uint32_t hero = game.compiled.root_value_slot;
  check(game.state_entry(first, villain, 0, 0) == 0,
        "first board starts with first node");
  check(game.state_entry(first, villain, 1, 0) == 1,
        "hands are contiguous within an action");
  check(game.state_entry(first, villain, 0, 1) == 32,
        "actions follow padded hand vectors");
  check(game.state_entry(first, hero, 0, 0) == 64,
        "nodes follow each other within a board");
  check(game.state_entry(first, hero, 1, 2) == 129,
        "hero actions also use padded hand vectors");
  check(game.state_entry(second, villain, 0, 0) == 160,
        "second board follows first board");
  check(game.state_entry(second, hero, 1, 2) == 289,
        "second board uses the same node layout");

  game.regrets[game.state_entry(first, hero, 0, 1)] = 5.0F;
  check(game.regrets[game.state_entry(second, hero, 0, 1)] == 0.0F,
        "boards have independent solver state");
  check(game.board_regrets(first).size() == 160,
        "board regret view covers one board");
  check(game.board_cumulative_strategy(second).size() == 160,
        "board strategy view covers one board");
  check(game.compiled.fold_payoffs == std::vector<float>{-1.0F},
        "compiled street retains fold terminal metadata");
  check(game.compiled.showdowns.size() == 2,
        "compiled street retains path-specific showdown metadata");
  check(game.compiled.showdown_value_count == 2,
        "compiled street counts path-specific showdowns");
  check(game.compiled.boundary_value_count == 1,
        "compiled street counts path-specific boundaries");
}

void test_fixed_flop_board_indexing() {
  RunoutIndex boards(make_mask(std::array<uint8_t, 3>{2, 5, 8}));

  check(boards.board_count(Street::Flop) == 1, "fixed flop has one board");
  check(boards.board_count(Street::Turn) == 49, "fixed flop has 49 turns");
  check(boards.board_count(Street::River) == 2352, "each turn has 48 rivers");
  check(boards.child_count(Street::Flop) == 49,
        "flop has one child per non-flop card");
  check(boards.child_count(Street::Turn) == 48,
        "turn has one child per non-board card");

  check(boards.child_board(Street::Flop, 0, 0) == 0,
        "first available turn card produces first turn");
  check(boards.child_board(Street::Flop, 0, 6) == 4,
        "turn index removes lower flop cards");

  BoardIndex turn = boards.child_board(Street::Flop, 0, 6);
  BoardIndex first_river = boards.child_board(Street::Turn, turn, 0);
  BoardIndex later_river = boards.child_board(Street::Turn, turn, 7);
  check(first_river == turn * 48, "each turn owns a contiguous river block");
  check(later_river == turn * 48 + 4,
        "river offset removes lower public cards");
  check(boards.turn_card(turn) == 6, "turn card is derivable from turn index");
  check(boards.river_card(later_river) == 7,
        "river card is derivable from river index");
  check(boards.board_mask(Street::Turn, turn) ==
            make_mask(std::array<uint8_t, 4>{2, 5, 6, 8}),
        "turn mask is derivable from turn index");
  check(boards.board_mask(Street::River, later_river) ==
            make_mask(std::array<uint8_t, 5>{2, 5, 6, 7, 8}),
        "river mask is derivable from river index");

  std::vector<bool> seen_turns(TURN_BOARD_COUNT);
  std::vector<bool> seen_rivers(RIVER_BOARD_COUNT);
  for (uint8_t turn_card = 0; turn_card < DECK_SIZE; ++turn_card) {
    if ((boards.board_mask(Street::Flop, 0) & card_mask(turn_card)) != 0) {
      continue;
    }
    BoardIndex turn_board = boards.child_board(Street::Flop, 0, turn_card);
    seen_turns[turn_board] = true;
    check(boards.turn_card(turn_board) == turn_card,
          "every turn transition round-trips");

    uint64_t turn_mask = boards.board_mask(Street::Turn, turn_board);
    for (uint8_t river_card = 0; river_card < DECK_SIZE; ++river_card) {
      if ((turn_mask & card_mask(river_card)) != 0) {
        continue;
      }
      BoardIndex river_board =
          boards.child_board(Street::Turn, turn_board, river_card);
      seen_rivers[river_board] = true;
      check(boards.river_card(river_board) == river_card,
            "every river transition round-trips");
    }
  }
  check(std::ranges::all_of(seen_turns, [](bool seen) { return seen; }),
        "direct indexing covers every turn exactly");
  check(std::ranges::all_of(seen_rivers, [](bool seen) { return seen; }),
        "direct indexing covers every river exactly");
}

} // namespace

int main() {
  try {
    test_shared_topology_and_child_order();
    test_board_node_action_padded_hand_layout();
    test_fixed_flop_board_indexing();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All street game tests passed\n";
  return EXIT_SUCCESS;
}
