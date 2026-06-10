#include "spmv_poker/game_tree.h"

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

void test_general_tree() {
  GameTree tree(1081);
  NodeIndex fold = tree.add_fold_node(-2.0F);
  NodeIndex showdown0 = tree.add_showdown_node(3, 4.0F, -2.0F);
  NodeIndex showdown1 = tree.add_showdown_node(7, 4.0F, -2.0F);

  std::array chance_children{
      GameEdge{.child = showdown0, .probability = 0.25F},
      GameEdge{.child = showdown1, .probability = 0.75F},
  };
  NodeIndex chance = tree.add_chance_node(chance_children);

  std::array decision_children{fold, chance};
  NodeIndex root = tree.add_decision_node(Player::Hero, decision_children);
  tree.root = root;

  check(tree.root == root, "tree stores root");
  check(tree.nodes.size() == 5, "tree stores every node");
  check(tree.edges.size() == 4, "tree stores every edge");
  check(tree.decisions.size() == 1, "tree stores decision state");

  const GameNode &root_node = tree.nodes[root];
  check(root_node.type == NodeType::Decision, "root is a decision");
  check(root_node.player == Player::Hero, "root stores acting player");
  check(root_node.decision_index == 0, "root maps to decision state");
  auto root_children = tree.children(root_node);
  check(root_children[0].child == fold, "action zero preserves child order");
  check(root_children[1].child == chance, "action one preserves child order");

  const GameNode &chance_node = tree.nodes[chance];
  auto runouts = tree.children(chance_node);
  check(runouts[0].probability == 0.25F, "chance probability zero");
  check(runouts[1].probability == 0.75F, "chance probability one");

  check(tree.nodes[fold].payoff == -2.0F, "fold stores payoff");
  check(tree.nodes[showdown1].runout_index == 7, "showdown stores runout");
  check(tree.nodes[showdown1].win_payoff == 4.0F,
        "showdown stores win payoff");
  check(tree.nodes[showdown1].loss_payoff == -2.0F,
        "showdown stores loss payoff");
}

void test_decision_offsets_survive_state_growth() {
  GameTree tree(2);
  NodeIndex fold = tree.add_fold_node(-1.0F);
  NodeIndex showdown = tree.add_showdown_node(0, 1.0F, -1.0F);
  std::array children{fold, showdown};
  tree.add_decision_node(Player::Villain, children);
  DecisionNode first = tree.decisions[0];

  for (size_t index = 0; index < 100; ++index) {
    tree.add_decision_node(Player::Hero, children);
  }
  tree.state.regret_span(first)[0] = 9.0F;

  check(tree.state.regrets[0] == 9.0F,
        "decision offsets survive global state growth");
}

void test_player_specific_decision_sizes() {
  GameTree tree(2, 3);
  NodeIndex fold = tree.add_fold_node(-1.0F);
  std::array children{fold, fold};

  tree.add_decision_node(Player::Hero, children);
  tree.add_decision_node(Player::Villain, children);

  check(tree.decisions[0].hand_count == 2,
        "hero decision uses hero hand count");
  check(tree.decisions[1].hand_count == 3,
        "villain decision uses villain hand count");
  check(tree.state.regrets.size() == 10,
        "decision state uses player-specific hand counts");
}

void test_deal_node_and_board_mask() {
  GameTree tree(2);
  NodeIndex first = tree.add_fold_node(1.0F);
  NodeIndex second = tree.add_fold_node(1.0F);
  std::array children{
      GameEdge{.child = first,
               .probability = 1.0F / 44.0F,
               .dealt_mask = card_mask(7)},
      GameEdge{.child = second,
               .probability = 1.0F / 44.0F,
               .dealt_mask = card_mask(8)},
  };
  tree.root = tree.add_card_deal_node(children);

  check(tree.nodes[tree.root].type == NodeType::Chance,
        "card deal uses chance node type");
  check(tree.nodes[tree.root].chance_kind == ChanceKind::CardDeal,
        "chance node stores card-deal semantics");
  check(tree.children(tree.nodes[tree.root])[0].dealt_mask == card_mask(7),
        "card-deal edge stores dealt card mask");
}

} // namespace

int main() {
  try {
    test_general_tree();
    test_decision_offsets_survive_state_growth();
    test_player_specific_decision_sizes();
    test_deal_node_and_board_mask();
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "All game tree tests passed\n";
  return EXIT_SUCCESS;
}
