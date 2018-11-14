#include <future>

#include "agent.hpp"


namespace hlt {

struct rollout_item {
    int state[64*64*17];
    int action[6];
    double value;
    double log_prob; //TODO: Change to Tensor
    double reward;
    int done;
};

Agent::Agent(Halite &game, int stateWidthHeight, int stateDepth, int actionSize)
: game(game)
{
}

void Agent::step(){
    return;
}

std::unique_ptr<hlt::Halite> reset_game(){
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
    auto game_statistics = std::make_unique<hlt::GameStatistics>();
    auto replay = std::make_unique<hlt::Replay>(*game_statistics, map_parameters.num_players, map_parameters.seed, map);

    auto game = std::make_unique<hlt::Halite>(map, *game_statistics, *replay);
    return game;
}



void Agent::generate_rollout() {
    std::vector<rollout_item> rollout;
    //TODO: Set up list of episode rewards
    //Reset environment for new game
    auto game_unique_ptr = reset_game();
    auto game_ptr = game_unique_ptr.get();

    auto grid = game_ptr->map.grid;

    return;
}

void Agent::process_rollout(){
    return;
}

void Agent::train_network(){
    return;
}

}