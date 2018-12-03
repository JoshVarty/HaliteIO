#ifndef HALITE_H
#define HALITE_H

#include "Store.hpp"
#include "mapgen/Generator.hpp"
#include <memory>

namespace hlt {

struct GameStatistics;

class HaliteImpl;

struct Replay;

/** Halite game interface, exposing the top level of the game. */
class Halite final {
    /** Transient game state. */


    /** Friend classes have full access to game state. */

    friend class HaliteImpl;

    std::unique_ptr<HaliteImpl> impl; /**< The pointer to implementation. */
    std::mt19937 rng;                 /** The random number generator used for tie breaking. */

public:
    /** External game state. */
    GameStatistics &game_statistics;  /**< The statistics of the game. */

    unsigned long turn_number{};      /**< The turn number. */
    //Replay &replay;                   /**< Replay instance to collect info for visualizer. */
    //PlayerLogs logs;                  /**< The player logs. */
    Store store;                      /**< The entity store. */
    Map &map;                         /**< The game map. */

    /**
     * Constructor for the main game.
     *
     * @param map The game map.
     * @param networking_config The networking configuration.
     * @param game_statistics The game statistics to use.
     * @param replay The game replay to use.
     */
    Halite(Map &map,
           GameStatistics &game_statistics,
           Replay &replay);

    /**
     * Run the game.
     * @param player_commands The list of player commands.
     */
    void run_game(int numPlayers);

    /** Default destructor is defined where HaliteImpl is complete. */
    ~Halite();

    void initialize_game(int numPlayers);

    void update_inspiration();
    
    /** Retrieve and process commands, and update the game state for the current turn. */
    void process_turn(std::map<long, std::vector<AgentCommand>> rawCommands);
    
    bool game_ended();
    
    void rank_players();
    
    void update_player_stats();

};

}

#endif
