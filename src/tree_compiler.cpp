#include "spmv_poker/tree_compiler.h"

#include <cassert>
#include <limits>
#include <vector>

namespace spmv_poker {

namespace {

uint32_t checked_index(size_t value) {
  assert(value < INVALID_COMPILED_INDEX);
  return static_cast<uint32_t>(value);
}

uint16_t checked_action_count(size_t value) {
  assert(value <= std::numeric_limits<uint16_t>::max());
  return static_cast<uint16_t>(value);
}

uint32_t player_offset(const StreetTopology &topology, NodeIndex node) {
  assert(topology.is_player(node));
  return checked_index(node - topology.player_begin());
}

struct EndpointLayout {
  std::vector<uint32_t> category_indices;
  std::vector<CompiledShowdown> showdowns;
  uint32_t boundary_count = 0;
};

EndpointLayout compile_endpoint_layout(const StreetTopology &topology) {
  EndpointLayout layout;
  layout.category_indices.resize(topology.endpoint_count);
  for (size_t endpoint = 0; endpoint < topology.endpoint_nodes.size();
       ++endpoint) {
    const StreetNode &node = topology.nodes[topology.endpoint_nodes[endpoint]];
    if (node.type == StreetNodeType::Showdown) {
      layout.category_indices[endpoint] =
          checked_index(layout.showdowns.size());
      layout.showdowns.push_back({
          .win_payoff = node.win_payoff,
          .loss_payoff = node.loss_payoff,
      });
    } else {
      assert(node.type == StreetNodeType::StreetBoundary);
      layout.category_indices[endpoint] = layout.boundary_count++;
    }
  }
  return layout;
}

CompiledForwardPlan compile_forward_plan(const StreetTopology &topology,
                                         const EndpointLayout &endpoints,
                                         Player propagated_player) {
  CompiledForwardPlan plan;
  auto player_nodes = topology.players();
  size_t player_count = player_nodes.size();
  std::vector<uint32_t> node_inputs(player_count);

  uint32_t temporary_reach_count = 0;
  for (size_t node_offset = 0; node_offset < player_count; ++node_offset) {
    const StreetNode &node = player_nodes[node_offset];
    if (node.player != propagated_player) {
      node_inputs[node_offset] = temporary_reach_count++;
    }
  }

  size_t player_index = static_cast<size_t>(propagated_player);
  uint32_t retained_reach_count =
      checked_index(topology.player_reach_counts[player_index]);
  uint32_t retained_reach_begin = temporary_reach_count;
  for (size_t node_offset = 0; node_offset < player_count; ++node_offset) {
    const StreetNode &node = player_nodes[node_offset];
    if (node.player == propagated_player) {
      node_inputs[node_offset] =
          retained_reach_begin + checked_index(node.reach_index);
    }
  }

  uint32_t fold_reach_begin = retained_reach_begin + retained_reach_count;
  uint32_t fold_reach_count = checked_index(topology.fold_payoffs.size());
  uint32_t showdown_reach_begin = fold_reach_begin + fold_reach_count;
  uint32_t showdown_reach_count = checked_index(endpoints.showdowns.size());
  uint32_t boundary_reach_begin = showdown_reach_begin + showdown_reach_count;
  uint32_t boundary_reach_count = endpoints.boundary_count;

  plan.nodes.reserve(player_count);
  plan.actions.reserve(topology.children.size() + topology.fold_payoffs.size());

  for (size_t node_offset = 0; node_offset < player_count; ++node_offset) {
    const StreetNode &node = player_nodes[node_offset];
    uint32_t first_action = checked_index(plan.actions.size());
    for (size_t action = 0; action < node.action_count(); ++action) {
      uint32_t destination;
      if (node.is_fold_action(action)) {
        destination = fold_reach_begin + checked_index(node.fold_index);
      } else {
        NodeIndex child = topology.child_for_action(node, action);
        if (topology.is_player(child)) {
          destination = node_inputs[player_offset(topology, child)];
        } else {
          size_t endpoint = topology.endpoint_for_action(node, action);
          destination = (topology.is_showdown(child) ? showdown_reach_begin
                                                     : boundary_reach_begin) +
                        endpoints.category_indices[endpoint];
        }
      }
      plan.actions.push_back({.destination_slot = destination,
                              .cfr_action_offset = checked_index(
                                  action * node.padded_hand_count)});
    }
    plan.nodes.push_back({
        .input_reach_slot = node_inputs[node_offset],
        .cfr_offset_within_board = checked_index(node.cfr_offset_within_board),
        .action_begin = first_action,
        .action_count = checked_action_count(node.action_count()),
        .player = node.player,
    });
  }

  plan.root_input_reach_slot =
      node_inputs[player_offset(topology, topology.root)];
  plan.temporary_reach_count = temporary_reach_count;
  plan.retained_reach_begin = retained_reach_begin;
  plan.retained_reach_count = retained_reach_count;
  plan.fold_reach_begin = fold_reach_begin;
  plan.fold_reach_count = fold_reach_count;
  plan.showdown_reach_begin = showdown_reach_begin;
  plan.showdown_reach_count = showdown_reach_count;
  plan.boundary_reach_begin = boundary_reach_begin;
  plan.boundary_reach_count = boundary_reach_count;
  plan.workspace_slot_count = boundary_reach_begin + boundary_reach_count;
  return plan;
}

} // namespace

CompiledStreet compile_street(const StreetTopology &topology) {
  assert(topology.is_player(topology.root));

  CompiledStreet compiled;
  EndpointLayout endpoints = compile_endpoint_layout(topology);
  auto player_nodes = topology.players();
  compiled.nodes.reserve(player_nodes.size());
  compiled.backward_actions.reserve(topology.children.size() +
                                    topology.fold_payoffs.size());

  compiled.fold_value_begin = checked_index(player_nodes.size());
  compiled.fold_value_count = checked_index(topology.fold_payoffs.size());
  compiled.showdown_value_begin =
      compiled.fold_value_begin + compiled.fold_value_count;
  compiled.showdown_value_count = checked_index(endpoints.showdowns.size());
  compiled.boundary_value_begin =
      compiled.showdown_value_begin + compiled.showdown_value_count;
  compiled.boundary_value_count = endpoints.boundary_count;
  compiled.value_workspace_slot_count =
      compiled.boundary_value_begin + compiled.boundary_value_count;

  for (size_t node_offset = 0; node_offset < player_nodes.size();
       ++node_offset) {
    const StreetNode &node = player_nodes[node_offset];
    uint32_t first_action = checked_index(compiled.backward_actions.size());
    for (size_t action = 0; action < node.action_count(); ++action) {
      uint32_t value_source;
      if (node.is_fold_action(action)) {
        value_source =
            compiled.fold_value_begin + checked_index(node.fold_index);
      } else {
        NodeIndex child = topology.child_for_action(node, action);
        if (topology.is_player(child)) {
          value_source = player_offset(topology, child);
        } else {
          size_t endpoint = topology.endpoint_for_action(node, action);
          value_source =
              (topology.is_showdown(child) ? compiled.showdown_value_begin
                                           : compiled.boundary_value_begin) +
              endpoints.category_indices[endpoint];
        }
      }
      compiled.backward_actions.push_back(
          {.source_slot = value_source,
           .cfr_action_offset =
               checked_index(action * node.padded_hand_count)});
    }
    compiled.nodes.push_back({
        .output_value_slot = checked_index(node_offset),
        .cfr_offset_within_board = checked_index(node.cfr_offset_within_board),
        .retained_reach_slot = checked_index(node.reach_index),
        .action_begin = first_action,
        .action_count = checked_action_count(node.action_count()),
        .player = node.player,
    });
  }

  compiled.forward_plans[static_cast<size_t>(Player::Hero)] =
      compile_forward_plan(topology, endpoints, Player::Hero);
  compiled.forward_plans[static_cast<size_t>(Player::Villain)] =
      compile_forward_plan(topology, endpoints, Player::Villain);
  compiled.root_value_slot = player_offset(topology, topology.root);
  compiled.retained_reach_counts = {
      checked_index(topology.player_reach_counts[0]),
      checked_index(topology.player_reach_counts[1]),
  };
  compiled.street = topology.street;
  compiled.hand_counts = {
      checked_index(topology.hand_count_for(Player::Hero)),
      checked_index(topology.hand_count_for(Player::Villain)),
  };
  compiled.padded_hand_counts = {
      checked_index(topology.padded_hand_count_for(Player::Hero)),
      checked_index(topology.padded_hand_count_for(Player::Villain)),
  };
  compiled.state_entries_per_board =
      checked_index(topology.state_entries_per_board);
  compiled.fold_payoffs = topology.fold_payoffs;
  compiled.showdowns = std::move(endpoints.showdowns);
  return compiled;
}

StreetTree::StreetTree(StreetTopology topology, uint64_t flop_mask)
    : compiled(compile_street(topology)), board_index(flop_mask),
      regrets(board_index.board_count(compiled.street) *
                  compiled.state_entries_per_board,
              0.0F),
      cumulative_strategy(board_index.board_count(compiled.street) *
                              compiled.state_entries_per_board,
                          0.0F) {}

size_t StreetTree::state_entry(BoardIndex board, uint32_t node_slot,
                               size_t hand, size_t action) const {
  assert(board < board_index.board_count(compiled.street));
  assert(node_slot < compiled.nodes.size());
  const CompiledBackwardNode &node = compiled.nodes[node_slot];
  size_t player_index = static_cast<size_t>(node.player);
  assert(hand < compiled.hand_counts[player_index]);
  assert(action < node.action_count);
  return static_cast<size_t>(board) * compiled.state_entries_per_board +
         node.cfr_offset_within_board +
         action * compiled.padded_hand_counts[player_index] + hand;
}

std::span<float> StreetTree::board_regrets(BoardIndex board) {
  assert(board < board_index.board_count(compiled.street));
  return std::span(regrets).subspan(static_cast<size_t>(board) *
                                        compiled.state_entries_per_board,
                                    compiled.state_entries_per_board);
}

std::span<const float> StreetTree::board_regrets(BoardIndex board) const {
  assert(board < board_index.board_count(compiled.street));
  return std::span(regrets).subspan(static_cast<size_t>(board) *
                                        compiled.state_entries_per_board,
                                    compiled.state_entries_per_board);
}

std::span<float> StreetTree::board_cumulative_strategy(BoardIndex board) {
  assert(board < board_index.board_count(compiled.street));
  return std::span(cumulative_strategy)
      .subspan(static_cast<size_t>(board) * compiled.state_entries_per_board,
               compiled.state_entries_per_board);
}

std::span<const float>
StreetTree::board_cumulative_strategy(BoardIndex board) const {
  assert(board < board_index.board_count(compiled.street));
  return std::span(cumulative_strategy)
      .subspan(static_cast<size_t>(board) * compiled.state_entries_per_board,
               compiled.state_entries_per_board);
}

} // namespace spmv_poker
