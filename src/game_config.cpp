#include "spmv_poker/game_config.h"

#include <array>
#include <cassert>
#include <vector>

namespace spmv_poker {

namespace {

NodeIndex add_river_betting_tree(GameTree &tree,
                                 const TerminalTables &terminals,
                                 const GameConfig &config,
                                 size_t runout_index) {
  assert(runout_index < terminals.runouts.size());
  assert(config.starting_pot > 0.0F);
  assert(!config.bet_sizes.empty());
  float showdown_payoff = 0.5F * config.starting_pot;

  NodeIndex checked_showdown = tree.add_showdown_node(
      runout_index, showdown_payoff, -showdown_payoff);

  std::vector<NodeIndex> villain_after_check_children{checked_showdown};
  std::vector<NodeIndex> hero_root_children;
  villain_after_check_children.reserve(config.bet_sizes.size() + 1);
  hero_root_children.reserve(config.bet_sizes.size() + 1);

  for (float bet_size : config.bet_sizes) {
    assert(bet_size > 0.0F);
    float called_payoff = showdown_payoff + bet_size * config.starting_pot;

    NodeIndex hero_bet_fold = tree.add_fold_node(showdown_payoff);
    NodeIndex hero_bet_call =
        tree.add_showdown_node(runout_index, called_payoff, -called_payoff);
    std::array response_children{hero_bet_fold, hero_bet_call};
    hero_root_children.push_back(
        tree.add_decision_node(Player::Villain, response_children));

    NodeIndex villain_bet_fold = tree.add_fold_node(-showdown_payoff);
    NodeIndex villain_bet_call =
        tree.add_showdown_node(runout_index, called_payoff, -called_payoff);
    std::array checked_response_children{villain_bet_fold, villain_bet_call};
    villain_after_check_children.push_back(
        tree.add_decision_node(Player::Hero, checked_response_children));
  }

  NodeIndex villain_after_check =
      tree.add_decision_node(Player::Villain, villain_after_check_children);
  hero_root_children.insert(hero_root_children.begin(), villain_after_check);
  return tree.add_decision_node(Player::Hero, hero_root_children);
}

} // namespace

GameTree build_game_tree(const TerminalTables &terminals,
                         const GameConfig &config) {
  GameTree tree(terminals.hands(Player::Hero).size(),
                terminals.hands(Player::Villain).size());

  if (config.starting_street == StartingStreet::River) {
    assert(terminals.runouts.size() == 1);
    tree.root = add_river_betting_tree(tree, terminals, config, 0);
    return tree;
  }

  assert(terminals.runouts.size() == DECK_SIZE - 4);
  std::vector<GameEdge> runouts;
  runouts.reserve(terminals.runouts.size());
  for (size_t runout = 0; runout < terminals.runouts.size(); ++runout) {
    NodeIndex child =
        add_river_betting_tree(tree, terminals, config, runout);
    uint8_t river = terminals.runouts[runout][4];
    runouts.push_back(GameEdge{.child = child,
                               .probability = 1.0F / 44.0F,
                               .dealt_mask = card_mask(river)});
  }
  tree.root = tree.add_card_deal_node(runouts);
  return tree;
}

} // namespace spmv_poker
