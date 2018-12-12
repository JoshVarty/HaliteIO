#ifndef TYPES_H
#define TYPES_H

#include <cstdlib>
#include <torch/torch.h>

const int NUMBER_OF_PLAYERS = 2;                //The number of players in the game

const float MAX_HALITE_ON_MAP = 1000;           //The maximum natural drop of halite on the map
const float MAX_HALITE_ON_SHIP = 1000;          //The maximum halite a ship can hold
const float MAX_SCORE_APPROXIMATE = 50000;      //A rough estimate of a "Max" score that we'll use for scaling our player's scores

const int NUMBER_OF_FRAMES = 12;                //The number of NxN input frames to our neural network
const int GAME_WIDTH = 32;                      //The number of NxN input frames to our neural network
const int GAME_HEIGHT = 32;                     //The number of NxN input frames to our neural network

struct StepResult {
    double meanScore;
    double meanSteps;
    double meanLoss;
};

struct Cell {
    //Map info
    float halite_on_ground = 0.0;
    //Ship info
    long shipOwner = -1;
    long shipId = -1;
    float halite_on_ship = 0.0;
    //Structure Info
    long structureOwner = -1;
    bool dropOffPresent = false;
    bool spawnPresent = false;
};

/*A compressed representation of the state of the game from the perspective of a ship at any moment in time*/
struct GameState {
    Cell position[GAME_HEIGHT][GAME_WIDTH] = {};
    float scores[NUMBER_OF_PLAYERS] = {};
    float steps_remaining = -1;

    int entityX = -1;
    int entityY = -1;
    float halite_on_ship;
    long playerId;
};

struct ModelOutput {
    at::Tensor action;
    at::Tensor log_prob;
    at::Tensor value;
    at::Tensor entropy;
};

struct RolloutItem {
    GameState state;
    long action;
    float value;
    float log_prob;
    float reward;
    int done;
    long playerId;
};

struct CompleteRolloutResult {
    std::vector<RolloutItem> rollouts;
    std::vector<RolloutItem> spawn_rollouts;
    std::vector<long> scores;
    std::vector<long> gameSteps;
};

struct ProcessedRolloutItem {
    GameState state;
    long action;
    float log_prob;
    float returns;
    float advantage;
};


#endif


