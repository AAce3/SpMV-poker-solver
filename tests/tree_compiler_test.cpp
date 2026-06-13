#include "spmv_poker/tree_compiler.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace spmv_poker;

void check(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

StreetTopology make_topology() {
  StreetTopology topology(Street::Turn, 3);
  NodeIndex showdown = topology.add_showdown_node(1.0F, -1.0F);
  NodeIndex boundary = topology.add_boundary_node();
  std::array villain_children{showdown, boundary};
  NodeIndex villain =
      topology.add_player_node(Player::Villain, villain_children, -1.0F);
  std::array hero_children{villain, showdown};
  topology.root = topology.add_player_node(Player::Hero, hero_children, -2.0F);
  return topology;
}

void test_backward_plan() {
  StreetTopology topology = make_topology();
  CompiledStreet compiled = compile_street(topology);

  check(compiled.nodes.size() == 2, "compiler emits only player nodes");
  check(compiled.backward_actions.size() == 6,
        "compiler emits every action once");
  check(compiled.root_value_slot == 1,
        "root maps to its player-node value slot");
  check(compiled.fold_value_begin == 2,
        "fold values follow internal node values");
  check(compiled.showdown_value_begin == 4, "showdowns follow fold values");
  check(compiled.boundary_value_begin == 6,
        "boundaries follow showdown values");
  check(compiled.value_workspace_slot_count == 7,
        "value workspace covers every source");
  check(compiled.padded_hand_counts[0] == 32 &&
            compiled.padded_hand_counts[1] == 32,
        "compiler records player hand-vector widths");
  check(compiled.state_entries_per_board == 192,
        "compiler records the board CFR-state stride");

  const CompiledBackwardNode &villain = compiled.nodes[0];
  const CompiledBackwardNode &hero = compiled.nodes[1];
  check(villain.action_begin == 0 && villain.action_count == 3,
        "villain action range is contiguous");
  check(hero.action_begin == 3 && hero.action_count == 3,
        "hero action range is contiguous");
  check(villain.cfr_offset_within_board == 0 &&
            hero.cfr_offset_within_board == 96,
        "nodes preserve board-relative CFR offsets");
  check(villain.retained_reach_slot == 0 && hero.retained_reach_slot == 0,
        "retained reach indices remain player-local");

  check(compiled.backward_actions[0].source_slot == 2,
        "villain fold resolves to fold value");
  check(compiled.backward_actions[1].source_slot == 4,
        "villain showdown resolves to showdown value");
  check(compiled.backward_actions[2].source_slot == 6,
        "villain boundary resolves to boundary value");
  check(compiled.backward_actions[3].source_slot == 3,
        "hero fold resolves to its fold value");
  check(compiled.backward_actions[4].source_slot == 0,
        "hero internal action resolves to child value");
  check(compiled.backward_actions[5].source_slot == 5,
        "path-specific showdown resolves to its showdown value");
  check(compiled.backward_actions[1].cfr_action_offset == 32 &&
            compiled.backward_actions[2].cfr_action_offset == 64,
        "action offsets identify padded hand vectors");
}

void test_player_specific_forward_plans() {
  StreetTopology topology = make_topology();
  CompiledStreet compiled = compile_street(topology);
  const CompiledForwardPlan &hero =
      compiled.forward_plans[static_cast<size_t>(Player::Hero)];
  const CompiledForwardPlan &villain =
      compiled.forward_plans[static_cast<size_t>(Player::Villain)];

  check(hero.temporary_reach_count == 1 && hero.retained_reach_begin == 1,
        "hero plan retains reaches only for hero nodes");
  check(hero.nodes[0].input_reach_slot == 0 &&
            hero.nodes[1].input_reach_slot == 1,
        "hero plan resolves temporary and retained node inputs");
  check(hero.root_input_reach_slot == 1,
        "hero root starts in retained storage");
  check(hero.actions[4].destination_slot == 0,
        "hero action resolves villain child to temporary reach storage");
  check(hero.actions[1].destination_slot == 4,
        "hero plan resolves showdown into showdown reach storage");
  check(hero.actions[2].destination_slot == 6,
        "hero plan resolves boundary into boundary reach storage");
  check(hero.actions[5].destination_slot == 5,
        "hero plan packs path-specific showdowns together");
  check(hero.fold_reach_begin == 2 && hero.showdown_reach_begin == 4 &&
            hero.boundary_reach_begin == 6 && hero.workspace_slot_count == 7,
        "hero forward workspace lays out every reach category");

  check(villain.temporary_reach_count == 1 && villain.retained_reach_begin == 1,
        "villain plan retains reaches only for villain nodes");
  check(villain.nodes[0].input_reach_slot == 1 &&
            villain.nodes[1].input_reach_slot == 0,
        "villain plan swaps retained and temporary node inputs");
  check(villain.root_input_reach_slot == 0,
        "villain root starts in temporary reach storage");
  check(villain.actions[4].destination_slot == 1,
        "villain action resolves villain child to retained storage");
}

} // namespace

int main() {
  try {
    test_backward_plan();
    test_player_specific_forward_plans();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tree compiler tests passed\n";
  return EXIT_SUCCESS;
}
