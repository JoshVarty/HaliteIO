#include <future>
#include "Halite.hpp"


namespace hlt {

class Agent {

    friend class Halite;

    Halite &game;
    double discount_rate = 0.99;        //Amount by which to discount future rewards
    double tau = 0.95;                  //
    int learningRounds = 10;            //number of optimization rounds for a single rollout
    int mini_batch_number = 32;         //batch size for optimization 
    double ppo_clip = 0.2;              //
    int gradient_clip = 5;              //Clip gradient to try to prevent unstable learning
    int maximum_timesteps = 1200;       //Maximum timesteps over which to generate a rollout

    void generate_rollout();

    void process_rollout();

    void train_network();

    void step();

public:

    /**
     * Construct Agent from state size and action size
     *
     */
    explicit Agent(Halite &game, int stateWidthHeight, int stateDepth, int actionSize);

};
}