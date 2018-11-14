#include <future>
#include "HaliteImpl.hpp"


namespace hlt {

class Agent {

    HaliteImpl game;

    void generate_rollout();

    void process_rollout();

    void train_network();

    void step();

public:

    /**
     * Construct Agent from state size and action size
     *
     */
    explicit Agent(HaliteImpl &game, int stateWidthHeight, int stateDepth, int actionSize);

};
}