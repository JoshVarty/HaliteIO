#include <future>

#include "agent.hpp"


namespace hlt {

Agent::Agent(HaliteImpl &game, int stateWidthHeight, int stateDepth, int actionSize)
: game(game)
{

}

void Agent::step(){
    return;
}

void Agent::generate_rollout() {
    return;
}

void Agent::process_rollout(){
    return;
}

void Agent::train_network(){
    return;
}

}