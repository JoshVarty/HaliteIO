#ifndef HALITE_H
#define HALITE_H

#include "PlayerLog.hpp"
#include "Store.hpp"
#include "mapgen/Generator.hpp"

namespace hlt {

struct GameStatistics;

class HaliteImpl;

struct Replay;

struct Snapshot;

/** Halite game interface, exposing the top level of the game. */
class Halite final {
    /** Transient game state. */

    /** External game state. */

    /** Friend classes have full access to game state. */

    friend class HaliteImpl;

    std::unique_ptr<HaliteImpl> impl; /**< The pointer to implementation. */
    std::mt19937 rng;                 /** The random number generator used for tie breaking. */

public:
    Store store;                      /**< The entity store. */
    PlayerLogs logs;                  /**< The player logs. */
    Map &map;                         /**< The game map. */
    GameStatistics &game_statistics;  /**< The statistics of the game. */
    Replay &replay;                   /**< Replay instance to collect info for visualizer. */
    unsigned long turn_number{};      /**< The turn number. */

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
     * @param numPlayers The number of players in the game
     * @param snapshot A snapshot of game state.
     */
    void run_game(int numPlayers, const Snapshot &snapshot);

    void initialize_game(int numPlayers);

    void update_inspiration();
    
    /** Retrieve and process commands, and update the game state for the current turn. */
    void process_turn(std::map<long, std::vector<AgentCommand>> rawCommands);
    
    bool game_ended();
    
    void rank_players();
    
    void update_player_stats();

    /** Generate a snapshot string from current game state. */
    std::string to_snapshot(const mapgen::MapParameters &map_parameters);

    /** Default destructor is defined where HaliteImpl is complete. */
    ~Halite();
};

}

#endif
