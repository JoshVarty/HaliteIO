#include <future>
#include <sstream>

#include "Halite.hpp"
#include "HaliteImpl.hpp"
#include "Player.hpp"
#include "Enumerated.hpp"
#include "Logging.hpp"
#include <random>


namespace hlt {

    


Agent::Agent() {
}

double Agent::step(){
    auto rollout = generate_rollout();
    auto processed_rollout = process_rollout(rollout);

    return 0.0;
}

std::vector<Frame> Agent::parseGridIntoSlices(long playerId, hlt::Halite &game) {

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

std::vector<Agent::rollout_item> Agent::generate_rollout() {
    std::vector<Agent::rollout_item> rollout;
    // //TODO: Set up list of episode rewards

    //Reset environment for new game
    int map_width = 64;
    int map_height = 64;
    int numPlayers = 2;
    hlt::mapgen::MapType type = mapgen::MapType::Fractal;
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

    //Initialize implementation
    auto impl = std::make_unique<HaliteImpl>(game);
    impl->initialize_game(numPlayers);

    const auto &constants = Constants::get();
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
        impl->update_inspiration();
        game.replay.full_frames.back().add_entities(game.store);

        //TODO: implement actual commands
        std::map<long, std::vector<AgentCommand>> commands;

        auto &players = game.store.players;
        for (auto playerPair : players) {


            auto frames = parseGridIntoSlices(0, game);

            auto id = playerPair.first.value;
            std::vector<AgentCommand> playerCommands;
            auto player = playerPair.second;

            for(auto entityPair : player.entities) {
                auto entityId = entityPair.first;
                auto location = entityPair.second;

                auto entity = game.store.get_entity(entityId);
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

        impl->process_turn(commands);

        // Add end of frame state.
        game.replay.full_frames.back().add_end_state(game.store);

        if (impl->game_ended()) {
            break;
        }
    }

    game.game_statistics.number_turns = game.turn_number;
    // Add state of entities at end of game.
    game.replay.full_frames.emplace_back();
    impl->update_inspiration();
    game.replay.full_frames.back().add_entities(game.store);
    impl->update_player_stats();
    game.replay.full_frames.back().add_end_state(game.store);

    impl->rank_players();
    Logging::log("Game has ended");
    Logging::set_turn_number(Logging::ended);
    game.logs.set_turn_number(PlayerLog::ended);
    for (const auto &[player_id, player] : game.store.players) {
        game.replay.players.find(player_id)->second.terminated = player.terminated;
    }

    //Save replays
    // While compilers like G++4.8 report C++11 compatibility, they do not
    // support std::put_time, so we have to use strftime instead.
    const auto time = std::time(nullptr);
    const auto localtime = std::localtime(&time);
    static constexpr size_t MAX_DATE_STRING_LENGTH = 25;
    char time_string[MAX_DATE_STRING_LENGTH];
    std::strftime(time_string, MAX_DATE_STRING_LENGTH, "%Y%m%d-%H%M%S%z", localtime);

    // Output gamefile. First try the replays folder; if that fails, just use the straight filename.
    std::stringstream filename_buf;
    filename_buf << "replay-" << std::string(time_string);
    filename_buf << "-" << replay.map_generator_seed;
    filename_buf << "-" << map.width;
    filename_buf << "-" << map.height << ".hlt";
    auto filename = filename_buf.str();
    std::string output_filename = replay_directory + filename;
    //results["replay"] = output_filename;
    bool enable_compression = true;
    try {
        replay.output(output_filename, enable_compression);
    } catch (std::runtime_error &e) {
        Logging::log("Error: could not write replay to directory " + replay_directory + ", falling back on current directory.", Logging::Level::Error);
        replay_directory = "./";
        output_filename = replay_directory + filename;
        replay.output(output_filename, enable_compression);
    }
    Logging::log("Opening a file at " + output_filename);


    for(auto row : game.map.grid){
        for (auto cell : row) {
            auto energy = cell.energy;
            auto x = cell.entity;
            auto y = cell.owner;

            if(y.value > -1){
                auto tempbreak = 45;
                auto temp = y.value;
            }
        }
    }
    
    auto grid = game.map.grid;

    return rollout;
}

std::vector<Agent::processed_rollout_item> Agent::process_rollout(std::vector<Agent::rollout_item> rollout) {
    std::vector<Agent::processed_rollout_item> processed_rollout;
    return processed_rollout;
}

void Agent::train_network(){
    return;
}

























    

/**
 * Constructor for the main game.
 *
 * @param map The game map.
 * @param networking_config The networking configuration.
 * @param players The list of players.
 * @param game_statistics The game statistics to use.
 * @param replay The game replay to use.
 */
Halite::Halite(Map &map,
               GameStatistics &game_statistics,
               Replay &replay) :
        map(map),
        game_statistics(game_statistics),
        replay(replay),

        //impl(std::make_unique<HaliteImpl>(*this)),
        rng(replay.map_generator_seed) {}

/**
 * Run the game.
 * @param numPlayers The number of players in the game
 */
void Halite::run_game(int numPlayers, const Snapshot &snapshot) {
    // impl->initialize_game(numPlayers);
    // impl->run_game();
}

std::string Halite::to_snapshot(const hlt::mapgen::MapParameters &map_parameters) {
    std::stringstream output;

    output << HALITE_VERSION << SNAPSHOT_FIELD_DELIMITER;

    output << map_parameters.type
           << SNAPSHOT_LIST_DELIMITER << map_parameters.width
           << SNAPSHOT_LIST_DELIMITER << map_parameters.height
           << SNAPSHOT_LIST_DELIMITER << map_parameters.num_players
           << SNAPSHOT_LIST_DELIMITER << map_parameters.seed
           << SNAPSHOT_FIELD_DELIMITER;

    for (const auto &row : map.grid) {
        for (const auto &cell : row) {
            output << cell.energy << SNAPSHOT_LIST_DELIMITER;
        }
    }
    output << SNAPSHOT_FIELD_DELIMITER;

    for (const auto&[player_id, player] : store.players) {
        output << player_id
               << SNAPSHOT_FIELD_DELIMITER << player.energy
               << SNAPSHOT_FIELD_DELIMITER
               << player.factory.x << SNAPSHOT_SUBFIELD_DELIMITER
               << player.factory.y << SNAPSHOT_LIST_DELIMITER;

        for (const auto &dropoff : player.dropoffs) {
            output << dropoff.id << SNAPSHOT_SUBFIELD_DELIMITER
                   << dropoff.location.x << SNAPSHOT_SUBFIELD_DELIMITER
                   << dropoff.location.y << SNAPSHOT_LIST_DELIMITER;
        }

        output << SNAPSHOT_FIELD_DELIMITER;

        for (const auto&[entity_id, entity_location] : player.entities) {
            const auto &entity = store.entities.at(entity_id);
            output << entity_id << SNAPSHOT_SUBFIELD_DELIMITER
                   << entity_location.x << SNAPSHOT_SUBFIELD_DELIMITER
                   << entity_location.y << SNAPSHOT_SUBFIELD_DELIMITER
                   << entity.energy << SNAPSHOT_LIST_DELIMITER;
        }

        output << SNAPSHOT_FIELD_DELIMITER;
    }

    return output.str();
}

/** Default destructor is defined where HaliteImpl is complete. */
Halite::~Halite() = default;

}
