#include "spmv_poker/spmv_evaluator.h"
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

void test_transition_graph_and_schedule() {
  RunoutIndex boards(make_mask(std::array<uint8_t, 3>{2, 5, 8}));
  CompiledTransitionGraph graph =
      compile_transition_graph(boards, Street::Turn);

  size_t parent_count = boards.board_count(Street::Turn);
  size_t child_per_parent = boards.child_count(Street::Turn);
  check(graph.parent_street == Street::Turn, "parent street is preserved");
  check(graph.child_street == Street::River, "child street is derived");
  check(graph.child_offsets.size() == parent_count + 1,
        "transition graph stores one offset per parent board");
  check(graph.child_boards.size() == parent_count * child_per_parent,
        "transition graph stores every legal child exactly once");
  check(graph.local_chance_weights.size() == graph.child_boards.size(),
        "transition graph stores one local weight per edge");
  check(graph.child_offsets.back() == graph.child_boards.size(),
        "transition offsets terminate at the edge count");
  check(graph.local_chance_weights.front() ==
            1.0F / static_cast<float>(child_per_parent),
        "transition graph preserves uniform chance weights");

  ExecutionSchedule schedule =
      build_execution_schedule(boards, Street::Turn, 11, 96);
  size_t total_turn_boards = 0;
  for (size_t group = 0; group < schedule.turn_groups.size(); ++group) {
    const TurnGroup &turn_group = schedule.turn_groups[group];
    check(turn_group.parent_board_count <= schedule.turn_board_capacity,
          "turn groups fit their capacity");
    check(turn_group.child_board_count ==
              turn_group.parent_board_count * child_per_parent,
          "turn group child count is derived from the parent count");
    total_turn_boards += turn_group.parent_board_count;

    const std::span<const RiverGroup> river_groups =
        schedule.river_groups_for(group);
    size_t total_river_boards = 0;
    size_t expected_local_begin = 0;
    for (const RiverGroup &river_group : river_groups) {
      check(river_group.local_parent_board_begin == expected_local_begin,
            "river groups use local parent board indexing");
      check(river_group.child_board_count <= schedule.river_board_capacity,
            "river groups fit their capacity");
      check(river_group.child_board_count ==
                river_group.parent_board_count * child_per_parent,
            "river group child count is derived from the parent count");
      total_river_boards += river_group.parent_board_count;
      expected_local_begin += river_group.parent_board_count;
    }
    check(total_river_boards == turn_group.parent_board_count,
          "river groups cover the full turn group");
  }
  check(total_turn_boards == parent_count, "turn groups cover all parents");
}

void test_bounded_workspace_sizes() {
  StreetTopology topology = make_topology();
  CompiledStreet compiled = compile_street(topology);

  CpuStreetWorkspace street_workspace;
  street_workspace.prepare(compiled, 7);
  check(street_workspace.board_capacity == 7,
        "street workspace records its board capacity");
  check(street_workspace.reaches[0].size() ==
            compiled.forward_plans[0].workspace_slot_count * 7 *
                compiled.padded_hand_counts[0],
        "street reach workspace is capacity bounded");
  check(street_workspace.values[1].size() ==
            compiled.value_workspace_slot_count * 7 *
                compiled.padded_hand_counts[1],
        "street value workspace is capacity bounded");

  CpuTransitionWorkspace transition_workspace;
  transition_workspace.prepare(3, 7, 11, compiled);
  check(transition_workspace.endpoint_capacity == 3,
        "transition workspace records endpoint capacity");
  check(transition_workspace.parent_board_capacity == 7,
        "transition workspace records parent capacity");
  check(transition_workspace.child_board_capacity == 11,
        "transition workspace records child capacity");
  check(transition_workspace.child_boards.size() == 11,
        "transition workspace bounds child boards");
  check(transition_workspace.child_root_reaches[0].size() ==
            3 * 11 * compiled.padded_hand_counts[0],
        "transition child reaches are capacity bounded");

  CpuBoundaryValueAccumulator accumulator;
  accumulator.prepare(3, 7, compiled);
  check(accumulator.endpoint_capacity == 3,
        "boundary accumulator records endpoint capacity");
  check(accumulator.parent_board_capacity == 7,
        "boundary accumulator records parent capacity");
  check(accumulator.values[1].size() ==
            3 * 7 * compiled.padded_hand_counts[1],
        "boundary accumulator is capacity bounded");
}

} // namespace

int main() {
  try {
    test_backward_plan();
    test_player_specific_forward_plans();
    test_transition_graph_and_schedule();
    test_bounded_workspace_sizes();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All tree compiler tests passed\n";
  return EXIT_SUCCESS;
}
