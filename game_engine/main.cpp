#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <iterator>
#include <vector>
#include <algorithm>

#include "Constants.hpp"
#include "Generator.hpp"
#include "Halite.hpp"
#include "Replay.hpp"
#include "Enumerated.hpp"

#include <torch/torch.h>

struct ModelOutput {
    at::Tensor action;
    at::Tensor log_prob;
    at::Tensor value;
};

struct RolloutItem {
    frame state;
    long action;
    float value;
    float log_prob;
    float reward;
    int done;
    int step;
    long playerId;
};

struct ProcessedRolloutItem {
    frame state;
    long action;
    float log_prob;
    float returns;
    float advantage;
};

class Batcher {
public:
int batchSize;
int numEntries;
std::vector<ProcessedRolloutItem> data;
std::vector<ProcessedRolloutItem>::iterator batchStart;
std::vector<ProcessedRolloutItem>::iterator batchEnd;

    Batcher(int batchSize, std::vector<ProcessedRolloutItem> data) {
        this->batchSize = batchSize;
        this->data = data;
        this->numEntries = data.size();
        this->reset();
    }

    void reset() {
        this->batchStart = data.begin();
        this->batchEnd = data.begin();
        advance(batchEnd, batchSize);
    }

    bool end() {
        return this->batchStart == this->data.end();
    }

    std::vector<ProcessedRolloutItem> next_batch(){
        std::vector<ProcessedRolloutItem> batch;
        batch.insert(batch.end(), batchStart, batchEnd);

        batchStart = batchEnd; 

        //Advance batchEnd to either the end of the next batch or
        //to the end of the data, whichever comes first
        for(int i = 0; i < batchSize; i++){
            batchEnd = batchEnd + 1;
            if(batchEnd == this->data.end()) {
                break;
            }
        }

        return batch;
    }

    void shuffle() {
        std::random_shuffle(this->data.begin(), this->data.end());
        this->reset();
    }
};

struct ActorCriticNetwork : torch::nn::Module {
public:

    torch::Device device;

    ActorCriticNetwork()
    :   conv1(torch::nn::Conv2dOptions(12, 32, /*kernel_size=*/3)),
        conv2(torch::nn::Conv2dOptions(32, 32, /*kernel_size=*/3)),
         fc1(32 * 60 * 60, 256),
         fc2(256, 6),           //Actor head
         fc3(256, 1),           //Critic head
         device(torch::Device(torch::kCPU))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);

        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU" << std::endl;
            device_type = torch::kCUDA;
        } else {
            std::cout << "Training on CPU" << std::endl;
            device_type = torch::kCPU;
        }

        device = torch::Device(device_type);
        this->to(device);
    }

    ModelOutput forward(torch::Tensor x, torch::Tensor selected_action) {
        x = x.to(this->device);
        x = conv1->forward(x);
        x = torch::relu(x);
        x = torch::relu(conv2->forward(x));
        x = x.view({-1, 32 * 60 * 60});
        x = torch::relu(fc1->forward(x));

        auto a = fc2->forward(x);
        auto action_probabilities = torch::softmax(a, /*dim=*/1);

        if(selected_action.numel() == 0) {
            //See:  https://github.com/pytorch/pytorch/blob/f79fb58744ba70970de652e46ea039b03e9ce9ff/torch/distributions/categorical.py#L110
            //      https://pytorch.org/cppdocs/api/function_namespaceat_1ac675eda9cae4819bc9311097af498b67.html?highlight=multinomial
            selected_action = action_probabilities.multinomial(1, true);
        }
        else {
            selected_action = selected_action.to(device);
        }


        auto log_prob = action_probabilities.gather(1, selected_action).log();
        auto value = fc3->forward(x);
        //Return action, log_prob, value
        ModelOutput output {selected_action, log_prob, value};
        return output;
  }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};

class Agent {
private:

std::string unitCommands[6] = {"N","E","S","W","still","construct"};

double discount_rate = 0.99;        //Amount by which to discount future rewards
double tau = 0.95;                  //
int learningRounds = 5;             //number of optimization rounds for a single rollout
std::size_t mini_batch_number = 32; //batch size for optimization
double ppo_clip = 0.2;              //
int gradient_clip = 5;              //Clip gradient to try to prevent unstable learning
int minimum_rollout_size = 1000;    //Minimum number of rollouts we accumulate before training the network

frame parseGridIntoSlices(long playerId, hlt::Halite &game) {

    int no_of_cols = 64;
    int no_of_rows = 64;
    int offset = 0;

    int numRows = game.map.grid.size();
    int totalSteps = 0;
    if (numRows == 64) {
        totalSteps = 501;
    }
    else if (numRows == 56) {
        totalSteps = 476;
    }
    else if (numRows == 48) {
        totalSteps = 451;
    }
    else if (numRows == 40) {
        totalSteps = 426;
    }
    else {
        totalSteps = 401;
    }

    //Board info
    frame myFrame;
    auto frameData = myFrame.state;
    auto halite_locations = frameData[0];
    auto steps_remaining = frameData[1];
    //My global info
    auto my_ships = frameData[2];
    auto my_ships_halite = frameData[3];
    auto my_dropoffs = frameData[4];
    auto my_score = frameData[5];
    //Enemy global info
    auto enemy_ships = frameData[6];
    auto enemy_ships_halite = frameData[7];
    auto enemy_dropoffs = frameData[8];
    auto enemy_score = frameData[9];

    int cellY = 0;
    for(auto row : game.map.grid) {
        int cellX = 0;
        for (auto cell: row) {
            auto x = offset + cellX;
            auto y = offset + cellY;

            //auto halite = cell.energy.value;
            halite_locations[y][x] = cell.energy;

            if(cell.entity.value != -1) {
                //There is a ship here
                auto entity = game.store.get_entity(cell.entity);

                if(entity.owner.value == playerId) {
                    my_ships_halite[y][x] = entity.energy;
                    my_ships[y][x] = 1;
                }
                else {
                    enemy_ships_halite[y][x] = entity.energy;
                    enemy_ships[y][x] = 2;
                }
            }

            cellX = cellX + 1;
        }
        cellY = cellY + 1;
    }

    for(auto playerPair : game.store.players) {

        auto player = playerPair.second;
        auto spawn = player.factory;
        auto x = offset + spawn.x;
        auto y = offset + spawn.y;

        if(player.id.value == playerId) {
            //We mark our spawn as a 'dropoff' because it can also be used as one
            my_dropoffs[y][x] = 1;
        }
        else {
            //We mark the enemy spawn as a 'dropoff' because it can also be used as one
            enemy_dropoffs[y][x] = 2;
        }

        for(auto dropoff : player.dropoffs) {
            if(player.id.value == playerId) {
                my_dropoffs[y][x] = 1;
            }
            else {
                //We mark the enemy spawn as a 'dropoff' because it can also be used as one
                enemy_dropoffs[y][x] = 2;
            }
        }

        // Player score
        auto score = player.energy;
        auto floatScore = (float)score;
        if(player.id.value == playerId) {
            for(int i = 0; i < 64; i++) {
                for(int j = 0; j < 64; j++){
                    my_score[i][j] = floatScore;
                }
            }
        }
        else {
            for(int i = 0; i < 64; i++) {
                for(int j = 0; j < 64; j++){
                    enemy_score[i][j] = floatScore;
                }
            }
        }

        //Steps remaining
        auto steps_remaining_value = totalSteps - game.turn_number + 1;
        if(player.id.value == playerId) {
            for(int i = 0; i < 64; i++) {
                for(int j = 0; j < 64; j++){
                    steps_remaining[i][j] = steps_remaining_value;
                }
            }
        }
    }

    return myFrame;
}

enum GameResult {
    Player1,
    Player2,
    Tie
};

GameResult getWinner(hlt::PlayerStatistics p1, hlt::PlayerStatistics other){
      if (p1.last_turn_alive == other.last_turn_alive) {
        auto turn_to_compare = p1.last_turn_alive;
        while (p1.turn_productions[turn_to_compare] == other.turn_productions[turn_to_compare]) {
            if (--turn_to_compare < 0) {
                // Players exactly tied on all turns
                return GameResult::Tie;
            }
        }

        if(p1.turn_productions[turn_to_compare] < other.turn_productions[turn_to_compare]){
            return GameResult::Player2;
        }
        else{
            return GameResult::Player1;
        }
    } else {
        if(p1.last_turn_alive < other.last_turn_alive){
            return GameResult::Player2;
        }
        else{
            return GameResult::Player1;
        }
    }
}

std::vector<RolloutItem> generate_rollouts() {

    std::vector<RolloutItem> rollouts;

    while(rollouts.size() < minimum_rollout_size) {

        std::unordered_map<long, std::vector<RolloutItem>> rolloutsForCurrentGame;

        //Reset environment for new game
        int map_width = 64;
        int map_height = 64;
        int numPlayers = 2;
        hlt::mapgen::MapType type = hlt::mapgen::MapType::Fractal;
        auto seed = static_cast<unsigned int>(time(nullptr));
        hlt::mapgen::MapParameters map_parameters{type, seed, map_width, map_height, numPlayers};
        hlt::Map map(map_width, map_height);
        hlt::mapgen::Generator::generate(map, map_parameters);
        hlt::GameStatistics game_statistics;
        hlt::Replay replay{game_statistics, map_parameters.num_players, map_parameters.seed, map};
        hlt::Halite game(map, game_statistics, replay);

        game.initialize_game(numPlayers);

        const auto &constants = hlt::Constants::get();

        game.turn_number = 1;
        while(true) {

            // Add state of entities at start of turn.
            // First, update inspiration flags, so they can be used for
            // movement/mining and so they are part of the replay.
            game.update_inspiration();

            std::map<long, std::vector<AgentCommand>> commands;

            auto &players = game.store.players;
            int offset = 0;

            std::unordered_map<long, RolloutItem> rolloutCurrentTurnByEntityId;

            for (auto playerPair : players) {

                auto id = playerPair.first.value;
                std::vector<AgentCommand> playerCommands;
                auto player = playerPair.second;

                for(auto entityPair : player.entities) {
                    auto entityId = entityPair.first;
                    auto location = entityPair.second;

                    auto frames = parseGridIntoSlices(0, game);

                    auto entity = game.store.get_entity(entityId);
                    int no_of_rows = 64;
                    int no_of_cols = 64;

                    //Zero out all cells except for where our current unit is
                    auto entityLocationFrame = frames.state[10];
                    for(int i = 0; i < 64; i++) {
                        for(int j = 0; j < 64; j++) {
                            entityLocationFrame[i][j] = 0;
                        }
                    }
                    entityLocationFrame[offset + location.y][offset + location.x] = 1;

                    //Set entire frame to the score of the current unit
                    auto entityEnergyFrame = frames.state[11];
                    float energy = entity.energy;
                    for(int i = 0; i < 64; i++) {
                        for(int j = 0; j < 64; j++) {
                            entityEnergyFrame[i][j] = energy;
                        }
                    }

                    //TODO: Ask the neural network what to do now?
                    auto state = torch::from_blob(frames.state, {12,64,64});

                    //frames.debug_print();
                    state = state.unsqueeze(0);
                    torch::Tensor emptyAction;
                    auto modelOutput = myModel.forward(state, emptyAction);
                    auto actionIndex = modelOutput.action.item<int64_t>();

                    //Create and story rollout
                    RolloutItem current_rollout;
                    current_rollout.state = frames;
                    current_rollout.value = modelOutput.value.item<float>();
                    current_rollout.action = actionIndex;
                    current_rollout.log_prob = modelOutput.log_prob.item<float>();
                    current_rollout.step = game.turn_number;
                    current_rollout.playerId = player.id.value;
                    current_rollout.reward = 0;
                    //This seems backwards but we represent "Done" as 0 and "Not done" as 1
                    current_rollout.done = 1;

                    rolloutCurrentTurnByEntityId[entityId.value] = current_rollout;

                    std::string command = unitCommands[actionIndex];
                    playerCommands.push_back(AgentCommand(entityId.value, command));
                }

                // auto energy = player.energy;
                //if self.game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
                auto factoryCell = game.map.grid[player.factory.y][player.factory.x];
                if(game.turn_number <= 200 && player.energy >= constants.NEW_ENTITY_ENERGY_COST && factoryCell.entity.value == -1) {
                    long factoryId = -1;
                    std::string command = "spawn";
                    playerCommands.push_back(AgentCommand(factoryId, command));
                }

                commands[id] = playerCommands;
            }

            game.process_turn(commands);

            //Add current rollouts to list
            for(auto rolloutKeyValue : rolloutCurrentTurnByEntityId) {
                auto entityId = rolloutKeyValue.first;
                auto rolloutItem = rolloutKeyValue.second;
                rolloutsForCurrentGame[entityId].push_back(rolloutItem);
            }

            game.turn_number = game.turn_number + 1;
            if (game.game_ended() || game.turn_number >= constants.MAX_TURNS) {
                std::cout << "Game ended in: " << game.turn_number << " turns" << std::endl;

                auto winningId = -1;
                auto winner = getWinner(game.game_statistics.player_statistics[0], game.game_statistics.player_statistics[1]);
                if (winner == GameResult::Player1) {
                    winningId = 0;
                }
                else if (winner == GameResult::Player2) {
                    winningId = 1;
                }

                std::cout << std::endl;
                std::cout << "Winner: " << winningId << std::endl;
                std::cout << std::endl;
                auto p1TurnProductions = game.game_statistics.player_statistics[0].turn_productions;
                auto p2TurnProductions = game.game_statistics.player_statistics[1].turn_productions;
                std::cout << "Player 1 total mined: " << p1TurnProductions[p1TurnProductions.size() - 1] << std::endl;
                std::cout << "Player 2 total mined: " << p2TurnProductions[p2TurnProductions.size() - 1] << std::endl;
                std::cout << std::endl;

                //If the game ended we have to correct the "rewards" and the "dones"
                for(auto rolloutKeyValue : rolloutsForCurrentGame) {
                    auto entityId = rolloutKeyValue.first;
                    auto entityRollout = rolloutKeyValue.second;
                    auto lastRolloutItem = entityRollout[entityRollout.size() - 1];

                    //This seems backwards but we represent "Done" as 0 and "Not done" as 1
                    lastRolloutItem.done = 0;
                    if(lastRolloutItem.playerId == winningId) {
                        lastRolloutItem.reward = 1;
                    }
                    else {
                        lastRolloutItem.reward = -1;
                    }

                    rolloutsForCurrentGame[entityId][entityRollout.size() - 1] = lastRolloutItem;
                    rolloutsForCurrentGame[entityId].push_back(lastRolloutItem);

                    rollouts.insert(rollouts.end(), rolloutsForCurrentGame[entityId].begin(), rolloutsForCurrentGame[entityId].end());
                }

                break;
            }
        }
    }

    return rollouts;
}

std::vector<ProcessedRolloutItem> process_rollouts(std::vector<RolloutItem> rollouts) {
    std::vector<ProcessedRolloutItem> processed_rollouts;

    //Get last value
    float advantage = 0;
    auto currentReturn = rollouts[rollouts.size() - 1].value;

    float advantage_mean;

    for(int i = rollouts.size() - 2;  i >= 0; i--) {
        auto rolloutItem = rollouts[i];
        auto nextValue = rollouts[i + 1].value;

        currentReturn = rolloutItem.reward + this->discount_rate * rolloutItem.done * currentReturn;
        auto td_error = currentReturn + this->discount_rate * rolloutItem.done * nextValue - rolloutItem.value;
        advantage = advantage * this->tau * this->discount_rate * rolloutItem.done + td_error;

        ProcessedRolloutItem processedRolloutItem;
        processedRolloutItem.state = rolloutItem.state;
        processedRolloutItem.action = rolloutItem.action;
        processedRolloutItem.log_prob = rolloutItem.log_prob;
        processedRolloutItem.returns = currentReturn;
        processedRolloutItem.advantage = advantage;
        processed_rollouts.push_back(processedRolloutItem);

        advantage_mean = advantage_mean + advantage;   //Accumulate all advantages
    }

    //Calculate mean from sum of advantages
    advantage_mean = advantage_mean / processed_rollouts.size();

    float advantage_std;
    for(int i = 0; i < processed_rollouts.size(); i++) {
        auto rolloutItem = processed_rollouts[i];
        auto differenceFromMean = (rolloutItem.advantage - advantage_mean);
        advantage_std = advantage_std + (differenceFromMean * differenceFromMean);
    }

    //Calculate std from sum of squared differences
    advantage_std = advantage_std / processed_rollouts.size();
    advantage_std = sqrt(advantage_std);

    //Normalize all of the advantages
    for(auto processedRolloutItem : processed_rollouts) {
        processedRolloutItem.advantage = (processedRolloutItem.advantage - advantage_mean) / advantage_std;
    }

    return processed_rollouts;
}

void train_network(std::vector<ProcessedRolloutItem> processed_rollout) {

    Batcher batcher(std::min(this->mini_batch_number, processed_rollout.size()), processed_rollout);
    for(int i = 0; i < this->learningRounds; i++) {
        //Shuffle the rollouts
        batcher.shuffle();
        frame sampled_states[this->mini_batch_number];
        float sampled_actions[this->mini_batch_number];
        float sampled_log_probs_old[this->mini_batch_number];
        float sampled_returns[this->mini_batch_number];
        float sampled_advantages[this->mini_batch_number];

        while(!batcher.end()) {
            auto nextBatch = batcher.next_batch();

            auto batchSize = (long)(nextBatch.size());

            for(int i = 0; i < batchSize; i++) {
                sampled_states[i] = nextBatch[i].state;
                sampled_actions[i] = nextBatch[i].action;
                sampled_log_probs_old[i] = nextBatch[i].log_prob;
                sampled_returns[i] = nextBatch[i].returns;
                sampled_advantages[i] = nextBatch[i].advantage;
            }

            //Create stack of Tensors as input to neural network
            std::vector<at::Tensor> stateList;
            for(int i = 0; i < batchSize; i++) {
                stateList.push_back(torch::from_blob(sampled_states[i].state, {12,64,64}));
            }

            auto batchInput = torch::stack(stateList);
            auto actionsTensor = torch::from_blob(sampled_actions, { batchSize });
            actionsTensor = actionsTensor.toType(torch::ScalarType::Long).unsqueeze(-1);
            auto modelOutput = this->myModel.forward(batchInput, actionsTensor);
            auto log_probs = modelOutput.log_prob;
            auto values = modelOutput.value;

            auto sampled_advantages_tensor = torch::from_blob(sampled_advantages, { batchSize, 1}).to(device);
            auto ratio = (log_probs - torch::from_blob(sampled_log_probs_old, { batchSize, 1}).to(device)).exp();
            auto obj = ratio * sampled_advantages_tensor;
            auto obj_clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * sampled_advantages_tensor;
            auto policy_loss = -torch::min(obj, obj_clipped).mean({0});

            // TODO: Why do they do 0.5?
            auto sampled_returns_tensor = torch::from_blob(sampled_returns, {batchSize, 1}).to(device);
            auto value_loss = 0.5 * (sampled_returns_tensor - values).pow(2).mean();

            optimizer.zero_grad();
            auto totalLoss = policy_loss + value_loss;
            totalLoss.backward();
            //TODO: Can we use gradient clipping?
            optimizer.step();
        }
    }

    std::cout << "Finished learning step" << std::endl;
}

public:

    ActorCriticNetwork myModel;
    torch::Device device;
    torch::optim::Adam optimizer;

    Agent():
        device(torch::Device(torch::kCPU)),
        optimizer(myModel.parameters(), torch::optim::AdamOptions(0.0000001))
    {
        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            device_type = torch::kCUDA;
        } else {
            device_type = torch::kCPU;
        }

        device = torch::Device(device_type);
    }

    double step() {
        auto rollouts = generate_rollouts();
        auto processed_rollout = process_rollouts(rollouts);
        // train_network
        train_network(processed_rollout);

        //TODO:
        //return average score?
        return 0.0;
    }
};



void ppo(Agent myAgent, uint numEpisodes) {
    std::vector<double> allScores;
    std::deque<double> lastHundredScores;

    for (uint i = 1; i < numEpisodes + 1; i++){

        double current_score = myAgent.step();
        allScores.push_back(current_score);
        //Keep track of the last 100 scores
        lastHundredScores.push_back(current_score);
        if(lastHundredScores.size() > 100) {
            lastHundredScores.pop_front();
        }

        double mean = 0;
        for (uint j = 0; j < lastHundredScores.size(); j++){
            mean += 0.01 * lastHundredScores[j];
        }

        if (i % 100 == 0) {
            //Every 100 episodes, display the mean reward
            std::cout << "Mean at step: " << i << ": " << mean;
        }

        //TODO: If network improves, save it.
    }
}



int main(int argc, char *argv[]) {

    Agent agent;
    ppo(agent, 500);

    return 0;
}
