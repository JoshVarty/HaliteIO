#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Constants.hpp"
#include "Generator.hpp"
#include "Halite.hpp"
#include "Logging.hpp"
#include "Replay.hpp"
#include "Snapshot.hpp"
#include "Enumerated.hpp"

#include <torch/torch.h>


class Agent {
private:

std::vector<Frame> parseGridIntoSlices(long playerId, hlt::Halite &game) {

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

    //TODO:
    // //My current unit info
    // std::vector<std::vector<float>> unit_halite;
    // std::vector<std::vector<float>> current_unit;
    // unit_halite.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    // current_unit.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));

    //Board info
    Frame halite_locations;
    Frame steps_remaining;
    //My global info
    Frame my_ships;
    Frame my_ships_halite;
    Frame my_dropoffs;
    Frame my_score;
    //Enemy global info
    Frame enemy_ships;
    Frame enemy_ships_halite;
    Frame enemy_dropoffs;
    std::vector<std::vector<float>> enemy_score;

    float initial_value = 0.0;
    halite_locations.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    steps_remaining.resize(no_of_rows, std::vector<float>(no_of_cols, totalSteps - game.turn_number));
    my_ships.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    my_ships_halite.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    my_dropoffs.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    enemy_ships.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    enemy_ships_halite.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
    enemy_dropoffs.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));

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
        if(player.id.value == playerId) {
            my_score.resize(no_of_rows, std::vector<float>(no_of_cols, score));
        }
        else {
            enemy_score.resize(no_of_rows, std::vector<float>(no_of_cols, score));
        }
    }

    std::vector<Frame> frame { halite_locations, steps_remaining, my_ships, my_ships_halite, my_dropoffs, my_score, enemy_ships, enemy_ships_halite, enemy_dropoffs, enemy_score};
    return frame;
}


std::vector<rollout_item> generate_rollout() {
    std::vector<rollout_item> rollout;
    // //TODO: Set up list of episode rewards

    //Reset environment for new game
    int map_width = 64;
    int map_height = 64;
    int numPlayers = 2;
    hlt::mapgen::MapType type = hlt::mapgen::MapType::Fractal;
    auto seed = static_cast<unsigned int>(time(nullptr));
    hlt::mapgen::MapParameters map_parameters{type, seed, map_width, map_height, numPlayers};
    hlt::Map map(map_width, map_height);
    hlt::mapgen::Generator::generate(map, map_parameters);
    std::string replay_directory = "replays/";
    constexpr auto SEPARATOR = '/';
    if (replay_directory.back() != SEPARATOR) replay_directory.push_back(SEPARATOR);
    hlt::GameStatistics game_statistics;
    hlt::Replay replay{game_statistics, map_parameters.num_players, map_parameters.seed, map};
    hlt::Halite game(map, game_statistics, replay);    

    game.initialize_game(numPlayers);

    const auto &constants = hlt::Constants::get();
    game.replay.players.insert(game.store.players.begin(), game.store.players.end());

    for (game.turn_number = 1; game.turn_number <= constants.MAX_TURNS; game.turn_number++) {

        Logging::set_turn_number(game.turn_number);
        game.logs.set_turn_number(game.turn_number);
        Logging::log([turn_number = game.turn_number]() {
            return "Starting turn " + std::to_string(turn_number);
        }, Logging::Level::Debug);
        
        // Create new turn struct for replay file, to be filled by further turn actions
        game.replay.full_frames.emplace_back();

        // Add state of entities at start of turn.
        // First, update inspiration flags, so they can be used for
        // movement/mining and so they are part of the replay.
        game.update_inspiration();
        game.replay.full_frames.back().add_entities(game.store);

        std::map<long, std::vector<AgentCommand>> commands;

        auto &players = game.store.players;
        int offset = 0;
        for (auto playerPair : players) {

            auto frames = parseGridIntoSlices(0, game);

            auto id = playerPair.first.value;
            std::vector<AgentCommand> playerCommands;
            auto player = playerPair.second;

            for(auto entityPair : player.entities) {
                auto entityId = entityPair.first;
                auto location = entityPair.second;

                auto entity = game.store.get_entity(entityId);
                int no_of_rows = 64;
                int no_of_cols = 64;
                float initial_value = 0.0;

                Frame entityLocation; 
                entityLocation.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
                entityLocation[offset + location.y][offset + location.x];
                
                Frame entityHalite; 
                entityHalite.resize(no_of_rows, std::vector<float>(no_of_cols, entity.energy));

                frames.push_back(entityLocation);
                frames.push_back(entityHalite);

                //TODO: Ask the neural network what to do now?

                std::string command = "";

                if(game.map.grid[location.y][location.x].energy >= 100 && !entity.energy >= 1000) {
                    //Mine (stay still)   
                    command = "stay";
                }
                else {
                    std::vector<std::string> moves {"N", "E", "S", "W"};
                    //Otherwise, take a random step
                    std::mt19937 rng;
                    rng.seed(std::random_device()());
                    std::uniform_int_distribution<std::mt19937::result_type> dist3(0,3); // distribution in range [0, 3]
                    auto randomIndex = dist3(rng);
                    command = moves[randomIndex];
                }
                playerCommands.push_back(AgentCommand(entityId.value, command));
            }

            // auto energy = player.energy;
            //if self.game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            auto factoryCell = game.map.grid[player.factory.y][player.factory.x];
            if(game.turn_number <= 200 && player.energy >= constants.NEW_ENTITY_ENERGY_COST && factoryCell.entity.value == -1){
                long factoryId = -1;
                std::string command = "spawn";
                playerCommands.push_back(AgentCommand(factoryId, command));
            }

            commands[id] = playerCommands;
        }

        game.process_turn(commands);

        // Add end of frame state.
        game.replay.full_frames.back().add_end_state(game.store);

        if (game.game_ended()) {
            break;
        }
    }

    //GAME IS OVER

    return rollout;
}

public:
    Agent() { }

    double step(){
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
    auto &constants = hlt::Constants::get_mut();

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    Agent agent;
    ppo(agent, 500);

    
    // Set the random seed
    auto seed = static_cast<unsigned int>(time(nullptr));

    // Use the seed to determine default map size
    std::mt19937 rng(seed);
    std::vector<hlt::dimension_type> map_sizes = {32, 40, 48, 56, 64};
    auto base_size = map_sizes[rng() % map_sizes.size()];
    constants.DEFAULT_MAP_WIDTH = constants.DEFAULT_MAP_HEIGHT = base_size;

    // Get the map parameters
    auto map_width = 64;
    auto map_height = 64;
    auto n_players = 2;

    auto verbosity = 3;
    Logging::set_level(static_cast<Logging::Level>(verbosity));

    hlt::mapgen::MapType type = hlt::mapgen::MapType::Fractal;
    hlt::mapgen::MapParameters map_parameters{type, seed, map_width, map_height, n_players};
    hlt::Snapshot snapshot;
    hlt::Map map(map_parameters.width, map_parameters.height);
    hlt::mapgen::Generator::generate(map, map_parameters);
    hlt::GameStatistics game_statistics;
    hlt::Replay replay{game_statistics, map_parameters.num_players, map_parameters.seed, map};
    hlt::Halite game(map, game_statistics, replay);
    game.run_game(n_players, snapshot);

    return 0;
}
