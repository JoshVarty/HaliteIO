#include <cstdlib>
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
#include "types.hpp"
#include "batcher.hpp"
#include "model.hpp"

#include <torch/torch.h>

class Agent {
private:

std::string unitCommands[6] = {"N","E","S","W","still","construct"};

Frame parseGridIntoSlices(long playerId, hlt::Halite &game) {

    int no_of_rows = GAME_HEIGHT;
    int no_of_cols = GAME_WIDTH;
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
    Frame myFrame;
    auto frameData = myFrame.state;
    auto halite_locations = frameData[0];   // Range from [-0.5, ~0.5]
    auto steps_remaining = frameData[1];    // Range from [-0.5, 0.5]
    //My global info
    auto my_ships = frameData[2];           // Range from [0,1]
    auto my_ships_halite = frameData[3];    // Range from [-0.5, 0.5]
    auto my_dropoffs = frameData[4];        // Range from [0, 1]
    auto my_score = frameData[5];           // Range from [-0.5, 0.5]
    //Enemy global info
    auto enemy_ships = frameData[6];        // Range from [0,1]
    auto enemy_ships_halite = frameData[7]; // Range from [-0.5, 0.5]
    auto enemy_dropoffs = frameData[8];     // Range from [0,1]
    auto enemy_score = frameData[9];        // Range from [-0.5, 0.5]

    int cellY = 0;
    for(auto row : game.map.grid) {
        int cellX = 0;
        for (auto cell: row) {
            auto x = offset + cellX;
            auto y = offset + cellY;

            auto scaled_halite = (cell.energy / MAX_HALITE_ON_MAP) - 0.5;
            halite_locations[y][x] = scaled_halite;

            if(cell.entity.value != -1) {
                //There is a ship here
                auto entity = game.store.get_entity(cell.entity);

                if(entity.owner.value == playerId) {
                    my_ships_halite[y][x] = (entity.energy / MAX_HALITE_ON_SHIP) - 0.5;
                    my_ships[y][x] = 1;
                }
                else {
                    enemy_ships_halite[y][x] = (entity.energy / MAX_HALITE_ON_SHIP) - 0.5;
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
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    my_score[i][j] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
                }
            }
        }
        else {
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    enemy_score[i][j] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
                }
            }
        }

        //Steps remaining
        auto steps_remaining_value = totalSteps - game.turn_number + 1;
        if(player.id.value == playerId) {
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    steps_remaining[i][j] = (steps_remaining_value / totalSteps) - 0.5; //We normalize between [-0.5, 0.5]
                }
            }
        }
    }

    return myFrame;
}

CompleteRolloutResult generate_rollouts() {

    CompleteRolloutResult result;
    std::vector<RolloutItem> rollouts;
    std::vector<RolloutItem> spawn_rollouts;
    std::vector<long> scores;
    std::vector<long> gameSteps;

    while(rollouts.size() < minimum_rollout_size) {

        std::unordered_map<long, std::vector<RolloutItem>> rolloutsForCurrentGame;
        std::unordered_map<long, std::vector<RolloutItem>> spawnRolloutsForCurrentGame;

        //Reset environment for new game
        long map_width = GAME_HEIGHT;
        long map_height = GAME_WIDTH;
        std::size_t numPlayers = 2;
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

                    //Zero out all cells except for where our current unit is
                    auto entityLocationFrame = frames.state[10];
                    for(int i = 0; i < GAME_HEIGHT; i++) {
                        for(int j = 0; j < GAME_WIDTH; j++) {
                            entityLocationFrame[i][j] = 0;
                        }
                    }
                    entityLocationFrame[offset + location.y][offset + location.x] = 1;

                    //Set entire frame to the score of the current unit
                    auto entityEnergyFrame = frames.state[11];
                    float energy = entity.energy;
                    for(int i = 0; i < GAME_HEIGHT; i++) {
                        for(int j = 0; j < GAME_WIDTH; j++) {
                            entityEnergyFrame[i][j] = (energy / MAX_HALITE_ON_SHIP) - 0.5;
                        }
                    }

                    //Ask the neural network what to do now?
                    auto state = torch::from_blob(frames.state, {NUMBER_OF_FRAMES, GAME_HEIGHT, GAME_WIDTH});

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
                    current_rollout.playerId = player.id.value;
                    current_rollout.reward = 0;
                    //This seems backwards but we represent "Done" as 0 and "Not done" as 1
                    current_rollout.done = 1;

                    rolloutCurrentTurnByEntityId[entityId.value] = current_rollout;

                    std::string command = unitCommands[actionIndex];
                    playerCommands.push_back(AgentCommand(entityId.value, command));
                }

                auto factoryCell = game.map.grid[player.factory.y][player.factory.x];
                if(player.energy >= constants.NEW_ENTITY_ENERGY_COST && factoryCell.entity.value == -1) {
                    long factoryId = -1;

                    auto frames = parseGridIntoSlices(0, game);

                    //Zero out all cells except for where the spawn is
                    auto entityLocationFrame = frames.state[10];
                    for(int i = 0; i < GAME_HEIGHT; i++) {
                        for(int j = 0; j < GAME_WIDTH; j++) {
                            entityLocationFrame[i][j] = 0;
                        }
                    }
                    entityLocationFrame[offset + player.factory.y][offset + player.factory.x] = 1;

                    //Set entire frame to zero as the spawn doesn't have any energy
                    auto entityEnergyFrame = frames.state[11];
                    for(int i = 0; i < GAME_HEIGHT; i++) {
                        for(int j = 0; j < GAME_WIDTH; j++) {
                            entityEnergyFrame[i][j] = 0;
                        }
                    }

                    //Ask the neural network what to do now?
                    auto state = torch::from_blob(frames.state, {NUMBER_OF_FRAMES, GAME_HEIGHT, GAME_WIDTH});
                    state = state.unsqueeze(0);
                    torch::Tensor emptyAction;
                    auto modelOutput = myModel.forward_spawn(state, emptyAction);

                    auto actionIndex = modelOutput.action.item<int64_t>();
                    //Create and story rollout
                    RolloutItem current_rollout;
                    current_rollout.state = frames;
                    current_rollout.value = modelOutput.value.item<float>();
                    current_rollout.action = actionIndex;
                    current_rollout.log_prob = modelOutput.log_prob.item<float>();
                    current_rollout.playerId = player.id.value;
                    current_rollout.reward = 0;
                    //This seems backwards but we represent "Done" as 0 and "Not done" as 1
                    current_rollout.done = 1;

                    if(actionIndex == 0) {
                        std::string command = "spawn";
                        playerCommands.push_back(AgentCommand(factoryId, command));
                    }

                    spawnRolloutsForCurrentGame[player.id.value].push_back(current_rollout);
                }

                commands[id] = playerCommands;
            }

            game.process_turn(commands);

            //Add current rollouts to list
            for(auto rolloutKeyValue : rolloutCurrentTurnByEntityId) {
                auto entityId = rolloutKeyValue.first;
                auto rolloutItem = rolloutKeyValue.second;

                //If this ship collided on this turn, penalize it
                if(std::find(game.store.selfCollidedEntities.begin(), game.store.selfCollidedEntities.end(), entityId) != game.store.selfCollidedEntities.end()) {
                    rolloutItem.reward = rolloutItem.reward - 0.05;
                }

                rolloutsForCurrentGame[entityId].push_back(rolloutItem);
            }

            game.turn_number = game.turn_number + 1;
            if (game.game_ended() || game.turn_number >= constants.MAX_TURNS) {
                //std::cout << "Game ended in: " << game.turn_number << " turns" << std::endl;
                gameSteps.push_back(game.turn_number);

                auto p1TurnProductions = game.game_statistics.player_statistics[0].turn_productions;
                auto p2TurnProductions = game.game_statistics.player_statistics[1].turn_productions;
                auto player1Score = p1TurnProductions[p1TurnProductions.size() - 1];
                auto player2Score = p2TurnProductions[p2TurnProductions.size() - 1];

                scores.push_back(player1Score);
                scores.push_back(player2Score);

                //If the game ended we have to correct the "rewards" and the "dones"
                for(auto rolloutKeyValue : rolloutsForCurrentGame) {
                    auto entityId = rolloutKeyValue.first;
                    auto entityRollout = rolloutKeyValue.second;
                    auto lastRolloutItem = entityRollout[entityRollout.size() - 1];

                    //This seems backwards but we represent "Done" as 0 and "Not done" as 1
                    lastRolloutItem.done = 0;
                    if(lastRolloutItem.playerId == 0) {
                        lastRolloutItem.reward = player1Score;
                    }
                    else {
                        lastRolloutItem.reward = player2Score;
                    }

                    rolloutsForCurrentGame[entityId][entityRollout.size() - 1] = lastRolloutItem;
                    rolloutsForCurrentGame[entityId].push_back(lastRolloutItem);

                    rollouts.insert(rollouts.end(), rolloutsForCurrentGame[entityId].begin(), rolloutsForCurrentGame[entityId].end());
                }



                break;
            }
        }
    }

    //Return scores along with rollouts
    result.rollouts = rollouts;
    result.scores = scores;
    result.gameSteps = gameSteps;
    result.spawn_rollouts = spawn_rollouts;
    return result;
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
        auto td_error = rolloutItem.reward + this->discount_rate * rolloutItem.done * nextValue - rolloutItem.value;
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
    for(std::size_t i = 0; i < processed_rollouts.size(); i++) {
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

std::vector<float> train_network(std::vector<ProcessedRolloutItem> processed_rollout) {

    std::vector<float> losses;
    Batcher batcher(std::min(this->mini_batch_number, processed_rollout.size()), processed_rollout);
    for(int i = 0; i < this->learningRounds; i++) {
        //Shuffle the rollouts
        batcher.shuffle();
        Frame sampled_states[this->mini_batch_number];
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
                stateList.push_back(torch::from_blob(sampled_states[i].state, {NUMBER_OF_FRAMES, GAME_HEIGHT, GAME_WIDTH}));
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
            auto policy_loss = -torch::min(obj, obj_clipped).mean();

            auto policy_loss_float = policy_loss.item<float>();

            // TODO: Why do they do 0.5?
            auto sampled_returns_tensor = torch::from_blob(sampled_returns, {batchSize, 1}).to(device);
            auto value_loss = 0.5 * (sampled_returns_tensor - values).pow(2).mean();
            auto value_loss_float = value_loss.item<float>();

            optimizer.zero_grad();
            auto totalLoss = policy_loss + value_loss;
            //std::cout << totalLoss << std::endl;
            totalLoss.backward();
            //TODO: Can we use gradient clipping?
            optimizer.step();

            //Keep track of losses
            losses.push_back(totalLoss.item<float_t>());
        }
    }

    //std::cout << "Finished learning step" << std::endl;
    return losses;
}

public:
    ActorCriticNetwork myModel;
    torch::Device device;

    float discount_rate;            //Amount by which to discount future rewards
    float tau;                      //
    int learningRounds;             //number of optimization rounds for a single rollout
    std::size_t mini_batch_number;  //batch size for optimization
    float ppo_clip;                 //Clip gradient to try to prevent unstable learning
    //int gradient_clip;
    std::size_t minimum_rollout_size;       //Minimum number of rollouts we accumulate before training the network
    float learning_rate;            //Rate at which the network learns
    
    torch::optim::Adam optimizer;

    Agent(float discount_rate, float tau, float learningRounds, float mini_batch_number, float ppo_clip, float minimum_rollout_size, float learning_rate):
        device(torch::Device(torch::kCUDA)),
        discount_rate(discount_rate),
        tau(tau),
        learningRounds(learningRounds),
        mini_batch_number(mini_batch_number),
        ppo_clip(ppo_clip),
        minimum_rollout_size(minimum_rollout_size),
        learning_rate(learning_rate),
        optimizer(myModel.parameters(), torch::optim::AdamOptions(learning_rate))
    {
        myModel.to(device);

        //Print out hyperparameter information
        std::cout << "discount_rate: " << discount_rate << std::endl;
        std::cout << "tau: " << tau << std::endl;
        std::cout << "learning_rounds: " << learningRounds << std::endl;
        std::cout << "mini_batch_number: " << mini_batch_number << std::endl;
        std::cout << "ppo_clip: " << ppo_clip << std::endl;
        //std::cout << "gradient_clip: " << gradient_clip << std::endl;
        std::cout << "minimum_rollout_size : " << minimum_rollout_size << std::endl;
        std::cout << "learning_rate: " << learning_rate << std::endl;
    }

    StepResult step() {
        std::vector<long> scores;
        std::vector<long> gameSteps;
        std::vector<float> losses;

        auto rolloutResult = generate_rollouts();
        scores.insert(scores.end(), rolloutResult.scores.begin(), rolloutResult.scores.end());
        gameSteps.insert(gameSteps.end(), rolloutResult.gameSteps.begin(), rolloutResult.gameSteps.end());
        
        auto processed_spawn_rollout = process_rollouts(rolloutResult.spawn_rollouts);
        auto spawnLosses = train_network(processed_spawn_rollout);

        auto processed_ship_rollout = process_rollouts(rolloutResult.rollouts);
        auto currentLosses = train_network(processed_ship_rollout);

        StepResult result;
        result.meanScore = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size(); 
        result.meanSteps = std::accumulate(gameSteps.begin(), gameSteps.end(), 0.0) / gameSteps.size();
        result.meanLoss = std::accumulate(currentLosses.begin(), currentLosses.end(), 0.0) / currentLosses.size();

        return result;
    }
};
