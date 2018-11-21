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

#include <torch/torch.h>


int main(int argc, char *argv[]) {
    auto &constants = hlt::Constants::get_mut();

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    
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
