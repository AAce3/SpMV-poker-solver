#pragma once

#include <cstdint>
#include <vector>

namespace spmv_poker {

enum class StartingStreet : uint8_t {
  Flop,
  Turn,
  River,
};

/**
 * Configuration for a no-raise betting tree.
 *
 * Bet sizes are fractions of the pot at the start of the river betting round.
 * A turn tree is currently simplified to deal the river immediately and then
 * use the configured river betting tree.
 */
struct GameConfig {
  StartingStreet starting_street = StartingStreet::River;
  float starting_pot = 2.0F;
  std::vector<float> bet_sizes{0.5F};
};

} // namespace spmv_poker
