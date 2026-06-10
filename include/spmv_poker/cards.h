#pragma once

#include <array>
#include <stddef.h>
#include <stdint.h>
#include <span>
#include <vector>

namespace spmv_poker {

constexpr size_t DECK_SIZE = 52;

enum class Player : uint8_t {
    Hero,
    Villain,
};

[[nodiscard]] constexpr Player opponent(Player player) {
    return player == Player::Hero ? Player::Villain : Player::Hero;
}

struct IndexRange {
    size_t begin;
    size_t count;

    template <typename T>
    [[nodiscard]] std::span<T> view(std::span<T> values) const {
        return values.subspan(begin, count);
    }

    template <typename T>
    [[nodiscard]] std::span<T> view(std::vector<T>& values) const {
        return std::span(values).subspan(begin, count);
    }

    template <typename T>
    [[nodiscard]] std::span<const T> view(const std::vector<T>& values) const {
        return std::span(values).subspan(begin, count);
    }
};

struct Hand {
    uint8_t first;
    uint8_t second;
    uint64_t mask;

    bool operator==(const Hand&) const = default;
};

[[nodiscard]] constexpr uint64_t card_mask(uint8_t card) {
    return uint64_t{1} << card;
}

template <size_t N>
[[nodiscard]] uint64_t make_mask(const std::array<uint8_t, N>& cards) {
    uint64_t mask = 0;
    for (uint8_t card : cards) {
        mask |= card_mask(card);
    }
    return mask;
}

[[nodiscard]] constexpr bool overlaps(const Hand& first, const Hand& second) {
    return (first.mask & second.mask) != 0;
}

}  // namespace spmv_poker
