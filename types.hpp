#ifndef TYPES_H
#define TYPES_H

#include <cstdlib>
#include <torch/torch.h>

const float MAX_HALITE_ON_MAP = 1000;           //The maximum natural drop of halite on the map
const float MAX_HALITE_ON_SHIP = 1000;          //The maximum halite a ship can hold
const float MAX_SCORE_APPROXIMATE = 50000;      //A rough estimate of a "Max" score that we'll use for scaling our player's scores

const int NUMBER_OF_FRAMES = 12;                //The number of NxN input frames to our neural network
const int GAME_WIDTH = 32;                //The number of NxN input frames to our neural network
const int GAME_HEIGHT = 32;                //The number of NxN input frames to our neural network

struct StepResult {
    double meanScore;
    double meanSteps;
    double meanLoss;
};

struct Frame {
public:
    float state[NUMBER_OF_FRAMES][GAME_HEIGHT][GAME_WIDTH] = {};

    void debug_print() {
        for(int i = 0; i < NUMBER_OF_FRAMES; i++){
            std::cout << std::endl << std::endl << "FRAME: " << i << std::endl;

            for(int j = 0; j < GAME_HEIGHT; j++) {
                std::cout << std::endl << j << ": \t";
                for(int k = 0; k < GAME_WIDTH; k++){
                    std::cout << state[i][j][k] << " ";
                }
            }
        }
    }
};

struct ModelOutput {
    at::Tensor action;
    at::Tensor log_prob;
    at::Tensor value;
};

struct RolloutItem {
    Frame state;
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
    Frame state;
    long action;
    float log_prob;
    float returns;
    float advantage;
};


#endif


