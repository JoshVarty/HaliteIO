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
    unsigned long turn_number{};      /**< The turn number. */

    /** External game state. */
    Map &map;                         /**< The game map. */
    GameStatistics &game_statistics;  /**< The statistics of the game. */
    Replay &replay;                   /**< Replay instance to collect info for visualizer. */

    /** Friend classes have full access to game state. */

    friend class HaliteImpl;

    friend class Agent;

    //std::unique_ptr<HaliteImpl> impl; /**< The pointer to implementation. */
    std::mt19937 rng;                 /** The random number generator used for tie breaking. */

public:
    Store store;                      /**< The entity store. */
    PlayerLogs logs;                  /**< The player logs. */

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
    void run_game(int numPlayers,
                  const Snapshot &snapshot);

    /** Generate a snapshot string from current game state. */
    std::string to_snapshot(const mapgen::MapParameters &map_parameters);

    /** Default destructor is defined where HaliteImpl is complete. */
    ~Halite();
};





class Agent {
private:
    double discount_rate = 0.99;        //Amount by which to discount future rewards
    double tau = 0.95;                  //
    int learningRounds = 10;            //number of optimization rounds for a single rollout
    int mini_batch_number = 32;         //batch size for optimization 
    double ppo_clip = 0.2;              //
    int gradient_clip = 5;              //Clip gradient to try to prevent unstable learning
    int maximum_timesteps = 1200;       //Maximum timesteps over which to generate a rollout


    struct rollout_item {
        int state[64*64*17];
        int action[6];
        double value;
        double log_prob; //TODO: Change to Tensor
        double reward;
        int done;
    };
    
    struct processed_rollout_item {
        int state[64*64*17];
        int action[6];
        double value;
        double log_prob; //TODO: Change to Tensor
        double reward;
        int done;
    };

    std::vector<rollout_item> generate_rollout();

    std::vector<processed_rollout_item> process_rollout(std::vector<rollout_item> rollout);

    void train_network();

public:

    /**
     * Construct Agent from state size and action size
     *
     */
    explicit Agent();

    double step();
};

}

#endif
