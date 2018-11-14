#include <future>
#include <sstream>

#include "Halite.hpp"
#include "HaliteImpl.hpp"
//
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

    hlt::Snapshot snapshot;
    game.impl->initialize_game(numPlayers, snapshot);

    game.impl->run_game();

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
        impl(std::make_unique<HaliteImpl>(*this)),
        rng(replay.map_generator_seed) {}

/**
 * Run the game.
 * @param numPlayers The number of players in the game
 */
void Halite::run_game(int numPlayers,
                      const Snapshot &snapshot) {
    impl->initialize_game(numPlayers, snapshot);
    impl->run_game();
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
