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
#include "../types.hpp"
#include "../batcher.hpp"
#include "../model.hpp"

#include <torch/torch.h>

class Agent {
private:

std::string unitCommands[6] = {"N","E","S","W","still","construct"};

torch::Tensor convertGameStateToTensor(GameState gameState) {

    auto playerId = gameState.playerId;
    //Halite Location
    auto halite_location = torch::tensor({GAME_HEIGHT, GAME_WIDTH});
    auto steps_remaining = torch::tensor({GAME_HEIGHT, GAME_WIDTH});
    // //My global info
    auto my_ships = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_ships_halite = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_dropoffs = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_score = torch::tensor({GAME_HEIGHT, GAME_WIDTH});
    //Enemy global info
    auto enemy_ships = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_ships_halite = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_dropoffs = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_score = torch::tensor({GAME_HEIGHT, GAME_WIDTH});

    //Ship specific information
    auto entity_location = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto entity_energy = torch::tensor({GAME_HEIGHT, GAME_WIDTH});
    entity_location[gameState.entityY][gameState.entityX] = 1;
    entity_energy[gameState.entityY][gameState.entityX] = gameState.halite_on_ship;

    steps_remaining.fill_(gameState.steps_remaining);

    //TODO: Generalize for more players
    if(playerId == 0) {
        my_score.fill_(gameState.scores[0]);
        enemy_score.fill_(gameState.scores[1]);
    } else {
        my_score.fill_(gameState.scores[1]);
        enemy_score.fill_(gameState.scores[0]);
    }

    for(std::size_t y = 0; y < GAME_HEIGHT; y++) {
        for(std::size_t x = 0; x < GAME_WIDTH; x++) {
            auto cell = gameState.position[y][x];
            halite_location[y][x] = cell.halite_on_ground;

            if(cell.shipId == playerId) {
                my_ships[y][x] = 1;
                my_ships_halite[y][x] = cell.halite_on_ship;
            }
            else if (cell.shipId != -1) {
                enemy_ships[y][x] = 1;
                enemy_ships_halite[y][x] = cell.halite_on_ship;
            }

            if(cell.structureOwner == playerId) {
                my_dropoffs[y][x] = 1;
            }
            else if (cell.structureOwner != -1) {
                enemy_dropoffs[y][x] = 1;
            }
        }
    }

    std::vector<torch::Tensor> frames {halite_location, steps_remaining,
    my_ships, my_ships_halite, my_dropoffs, my_score,
    enemy_ships, enemy_ships_halite, enemy_dropoffs, enemy_score,
    entity_location, entity_energy};

    auto stateTensor = torch::stack(frames);
    return stateTensor;
}

GameState parseGameIntoGameState(long playerId, hlt::Halite &game, int entityY, int entityX, float entityEnergy) {

    GameState gameState;

    gameState.entityY = entityY;
    gameState.entityX = entityX;
    gameState.halite_on_ship = entityEnergy;
    gameState.playerId = playerId;

    int no_of_rows = GAME_HEIGHT;
    int no_of_cols = GAME_WIDTH;

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

    int cellY = 0;
    for(auto row : game.map.grid) {
        int cellX = 0;
        for (auto cell: row) {
            auto x = cellX;
            auto y = cellY;

            float scaled_halite = (cell.energy / MAX_HALITE_ON_MAP) - 0.5;
            gameState.position[y][x].halite_on_ground = scaled_halite;

            if(cell.entity.value != -1) {
                //There is a ship here
                auto entity = game.store.get_entity(cell.entity);

                gameState.position[y][x].halite_on_ship = (entity.energy / MAX_HALITE_ON_SHIP) - 0.5;
                gameState.position[y][x].shipId = entity.id.value;
                gameState.position[y][x].shipOwner = entity.owner.value;;
            }

            cellX = cellX + 1;
        }
        cellY = cellY + 1;
    }


    for(auto playerPair : game.store.players) {

        auto player = playerPair.second;
        auto spawn = player.factory;

        //We consider spawn/factories to be both dropoffs and spawns
        gameState.position[spawn.y][spawn.x].dropOffPresent = true;
        gameState.position[spawn.y][spawn.x].spawnPresent = true;
        gameState.position[spawn.y][spawn.x].structureOwner = player.id.value;

        for(auto dropoff : player.dropoffs) {
            gameState.position[dropoff.location.y][dropoff.location.x].dropOffPresent = true;
            gameState.position[dropoff.location.y][dropoff.location.x].structureOwner = player.id.value;
        }

        // Player score
        auto score = player.energy;
        auto floatScore = (float)score;
        gameState.scores[player.id.value] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
    }

    //Steps remaining
    auto steps_remaining_value = totalSteps - game.turn_number + 1;
    auto scaled_steps = (steps_remaining_value / totalSteps) - 0.5; //We normalize between [-0.5, 0.5]
    gameState.steps_remaining = scaled_steps;

    return gameState;
}


CompleteRolloutResult generate_rollouts() {

    CompleteRolloutResult result;
    std::vector<RolloutItem> rollouts;
    std::vector<RolloutItem> spawn_rollouts;
    std::vector<long> scores;
    std::vector<long> gameSteps;
    const auto &constants = hlt::Constants::get();

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

                auto playerId = playerPair.first.value;
                std::vector<AgentCommand> playerCommands;
                auto player = playerPair.second;

                for(auto entityPair : player.entities) {
                    auto entityId = entityPair.first;
                    auto location = entityPair.second;

                    auto entity = game.store.get_entity(entityId);
                    auto gameState = parseGameIntoGameState(playerId, game, location.y, location.x, entity.energy);

                    //Ask the neural network what to do now?
                    //auto state = torch::from_blob(frames.state, {NUMBER_OF_FRAMES, GAME_HEIGHT, GAME_WIDTH});
                    auto state = convertGameStateToTensor(gameState);

                    //frames.debug_print();
                    state = state.unsqueeze(0);
                    torch::Tensor emptyAction;
                    auto modelOutput = myModel.forward(state, emptyAction);
                    auto actionIndex = modelOutput.action.item<int64_t>();

                    //Create and story rollout
                    RolloutItem current_rollout;
                    current_rollout.state = gameState;
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

                    auto gameState = parseGameIntoGameState(playerId, game, player.factory.y, player.factory.x, player.energy);
                    auto state = convertGameStateToTensor(gameState);
                    state = state.unsqueeze(0);
                    //Ask the neural network what to do now?
                    torch::Tensor emptyAction;
                    auto modelOutput = myModel.forward_spawn(state, emptyAction);

                    auto actionIndex = modelOutput.action.item<int64_t>();
                    //Create and story rollout
                    RolloutItem current_rollout;
                    current_rollout.state = gameState;
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

                commands[playerId] = playerCommands;
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

                //Need to do the same thing for spawnRollouts
                for(auto rolloutKeyValue : spawnRolloutsForCurrentGame) {
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

                    spawnRolloutsForCurrentGame[entityId][entityRollout.size() - 1] = lastRolloutItem;
                    spawnRolloutsForCurrentGame[entityId].push_back(lastRolloutItem);

                    spawn_rollouts.insert(spawn_rollouts.end(), spawnRolloutsForCurrentGame[entityId].begin(), spawnRolloutsForCurrentGame[entityId].end());
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

std::vector<float> train_network(std::vector<ProcessedRolloutItem> processed_rollout, bool spawnRollout) {

    std::vector<float> losses;
    Batcher batcher(std::min(this->mini_batch_number, processed_rollout.size()), processed_rollout);
    for(int i = 0; i < this->learningRounds; i++) {
        //Shuffle the rollouts
        batcher.shuffle();

        float sampled_actions[this->mini_batch_number];
        float sampled_log_probs_old[this->mini_batch_number];
        float sampled_returns[this->mini_batch_number];
        float sampled_advantages[this->mini_batch_number];

        while(!batcher.end()) {
            auto nextBatch = batcher.next_batch();
            auto batchSize = (long)(nextBatch.size());

            for(int i = 0; i < batchSize; i++) {
                sampled_actions[i] = nextBatch[i].action;
                sampled_log_probs_old[i] = nextBatch[i].log_prob;
                sampled_returns[i] = nextBatch[i].returns;
                sampled_advantages[i] = nextBatch[i].advantage;
            }

            //Create stack of Tensors as input to neural network
            std::vector<at::Tensor> stateList;
            for(int i = 0; i < batchSize; i++) {
                auto gameState = nextBatch[i].state;
                auto stateTensor = convertGameStateToTensor(gameState);
                stateList.push_back(stateTensor);
            }

            auto batchInput = torch::stack(stateList);
            auto actionsTensor = torch::from_blob(sampled_actions, { batchSize });
            actionsTensor = actionsTensor.toType(torch::ScalarType::Long).unsqueeze(-1);
            ModelOutput modelOutput;

            if(spawnRollout) {
                modelOutput = this->myModel.forward_spawn(batchInput, actionsTensor);
            }
            else{
                modelOutput = this->myModel.forward(batchInput, actionsTensor);
            }
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
            //losses.push_back(totalLoss.item<float_t>());
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
        myModel(true),
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

        std::cout << "About to process spawn rollout" << std::endl;
        auto processed_spawn_rollout = process_rollouts(rolloutResult.spawn_rollouts);
        std::cout << "Processed spawn rollout" << std::endl;
        //NOT TRAINING THIS PROPERLY BECAUSE YOURE NOT USING FORWARD_SPAWN
        std::cout << "About to train spawn rollout" << std::endl;
        auto spawnLosses = train_network(processed_spawn_rollout, true);
        std::cout << "Trained spawn rollout" << std::endl;

        auto processed_ship_rollout = process_rollouts(rolloutResult.rollouts);
        auto currentLosses = train_network(processed_ship_rollout, false);

        StepResult result;
        result.meanScore = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size(); 
        result.meanSteps = std::accumulate(gameSteps.begin(), gameSteps.end(), 0.0) / gameSteps.size();
        result.meanLoss = std::accumulate(currentLosses.begin(), currentLosses.end(), 0.0) / currentLosses.size();

        return result;
    }
};
