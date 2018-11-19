
#include <vector>
#include "Enumerated.hpp"
#include "Halite.hpp"


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
    
    std::vector<Frame> parseGridIntoSlices(long playerId, hlt::Halite &game);


public:

    /**
     * Construct Agent from state size and action size
     *
     */
    explicit Agent();

    double step();
};