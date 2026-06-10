cmake_minimum_required(VERSION 3.24)

project(spmv_poker_solver LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

foreach(option
    BUILD_CARD5
    BUILD_CARD6
    BUILD_PLO4
    BUILD_PLO5
    BUILD_PLO6
    BUILD_TESTS
    BUILD_EXAMPLES
)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()

set(BUILD_CARD7 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    poker_hand_evaluator
    GIT_REPOSITORY https://github.com/HenryRLee/PokerHandEvaluator.git
    GIT_TAG v0.5.3.1
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cpp
)
FetchContent_MakeAvailable(poker_hand_evaluator)

add_library(poker_core
    src/decision_node.cpp
    src/evaluator.cpp
    src/game_config.cpp
    src/game_tree.cpp
    src/range.cpp
    src/terminal.cpp
)
target_include_directories(poker_core PUBLIC include)
target_link_libraries(poker_core PRIVATE pheval7)

add_executable(poker_solver src/main.cpp)
target_link_libraries(poker_solver PRIVATE poker_core)

enable_testing()

add_executable(terminal_tests tests/terminal_test.cpp)
target_link_libraries(terminal_tests PRIVATE poker_core)
add_test(NAME terminal_tests COMMAND terminal_tests)

add_executable(decision_node_tests tests/decision_node_test.cpp)
target_link_libraries(decision_node_tests PRIVATE poker_core)
add_test(NAME decision_node_tests COMMAND decision_node_tests)

add_executable(game_tree_tests tests/game_tree_test.cpp)
target_link_libraries(game_tree_tests PRIVATE poker_core)
add_test(NAME game_tree_tests COMMAND game_tree_tests)

add_executable(evaluator_tests tests/evaluator_test.cpp)
target_link_libraries(evaluator_tests PRIVATE poker_core)
add_test(NAME evaluator_tests COMMAND evaluator_tests)
