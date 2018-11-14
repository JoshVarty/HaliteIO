#include <future>

#include "agent.hpp"


namespace hlt {

struct rollout_item {
    int state[64*64*17];
    int action[6];
    double value;
    double log_prob; //TODO: Change to Tensor
    double reward;
    int done;
};

Agent::Agent(HaliteImpl &game, int stateWidthHeight, int stateDepth, int actionSize)
: game(game)
{

}

void Agent::step(){
    return;
}

void Agent::generate_rollout() {
    std::vector<rollout_item> rollout;
    //TODO: Set up list of episode rewards
    //TODO: Reset environment for new game

    return;
}

void Agent::process_rollout(){
    return;
}

void Agent::train_network(){
    return;
}

}