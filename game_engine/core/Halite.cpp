#include <future>
#include <sstream>

#include "Halite.hpp"
#include "HaliteImpl.hpp"
#include "Player.hpp"
#include "Enumerated.hpp"
#include "Logging.hpp"

namespace hlt {

    


Agent::Agent()
{
}

double Agent::step(){
    auto rollout = generate_rollout();
    auto processed_rollout = process_rollout(rollout);

    return 0.0;
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

    auto impl = std::make_unique<HaliteImpl>(game);
    impl->initialize_game(numPlayers);

    //TODO: Move any initialization in run_game here
    //impl->run_game();

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
        std::map<uint, std::vector<std::string>> commands;

        auto &players = game.store.players;
        for (auto playerPair : players) {

            auto id = playerPair.first.value;
            std::vector<std::string> playerCommands;
            auto player = playerPair.second;

            // auto energy = player.energy;
            //if self.game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            auto cell = game.map.grid[player.factory.y][player.factory.x];

            if(game.turn_number <= 200 && player.energy >= constants.NEW_ENTITY_ENERGY_COST && cell.entity.value == -1){
                std::string command = "spawn";
                playerCommands.push_back(command);
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
        if (!player.terminated) {
            //TODO: Kill player locally
            //game.networking.kill_player(player);
        }
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
