#include <future>
#include <sstream>

#include "Halite.hpp"
#include "HaliteImpl.hpp"

namespace hlt {

/**
 * Constructor for the main game.
 *
 * @param map The game map.
 * @param networking_config The networking configuration.
 * @param players The list of players.
 * @param game_statistics The game statistics to use.
 * @param replay The game replay to use.
 */
Halite::Halite(Map &map,
               GameStatistics &game_statistics,
               Replay &replay) :
        map(map),
        game_statistics(game_statistics),
        replay(replay),
        impl(std::make_unique<HaliteImpl>(*this)),
        rng(replay.map_generator_seed) {}

/**
 * Run the game.
 * @param player_commands The list of player commands.
 */
void Halite::run_game(int n_players) {
    // impl->initialize_game(n_players, snapshot);
    // impl->run_game();
}

void Halite::initialize_game(int numPlayers){
    impl->initialize_game(numPlayers);
}
    
void Halite::update_inspiration() {
    impl->update_inspiration(); 
}

/** Retrieve and process commands, and update the game state for the current turn. */
void Halite::process_turn(std::map<long, std::vector<AgentCommand>> rawCommands){
    impl->process_turn(rawCommands);
}

bool Halite::game_ended() {
    return impl->game_ended();
}

void Halite::update_player_stats(){
    impl->update_player_stats();
}

void Halite::rank_players(){
    impl->rank_players();
}

/** Default destructor is defined where HaliteImpl is complete. */
Halite::~Halite() = default;

}
