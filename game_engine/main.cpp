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

/** The platform-specific path separator. */
#ifdef _WIN32
constexpr auto SEPARATOR = '\\';
#else
constexpr auto SEPARATOR = '/';
#endif

constexpr auto JSON_INDENT_LEVEL = 4;

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

    // if (json_results_switch.getValue()) {
    //     Logging::set_enabled(false);
    // }

    n_players = 2;
    // Read the player bot commands
    // auto bot_commands = command_args.getValue();
    // if (bot_commands.size() > constants.MAX_PLAYERS) {
    //     Logging::log("Too many players (max is " + std::to_string(constants.MAX_PLAYERS) + ")", Logging::Level::Error);
    //     return 1;
    // } else if (bot_commands.size() > n_players) {
    //     n_players = bot_commands.size();
    //     if (players_arg.isSet()) {
    //         Logging::log("Overriding the specified number of players", Logging::Level::Warning);
    //     }
    // }

    hlt::mapgen::MapType type = hlt::mapgen::MapType::Fractal;
    //std::istringstream type_stream(map_type_arg.getValue());
    //type_stream >> type;
    hlt::mapgen::MapParameters map_parameters{type, seed, map_width, map_height, n_players};

    hlt::Snapshot snapshot;
    // if (!snapshot_arg.getValue().empty()) {
    //     try {
    //         snapshot = hlt::Snapshot::from_str(snapshot_arg.getValue());
    //     } catch (const SnapshotError &err) {
    //         std::cerr << err.what() << std::endl;
    //         return 1;
    //     }
    //     map_parameters = snapshot.map_param;
    // }

    // net::NetworkingConfig networking_config{};
    // networking_config.ignore_timeout = timeout_switch.getValue();

    hlt::Map map(map_parameters.width, map_parameters.height);
    hlt::mapgen::Generator::generate(map, map_parameters);

    std::string replay_directory = "replays/";
    if (replay_directory.back() != SEPARATOR) replay_directory.push_back(SEPARATOR);

    hlt::GameStatistics game_statistics;
    hlt::Replay replay{game_statistics, map_parameters.num_players, map_parameters.seed, map};
    Logging::log("Map seed is " + std::to_string(map_parameters.seed));

    hlt::Halite game(map, game_statistics, replay);
    game.run_game(n_players, snapshot);

    // const auto &overrides = override_args.getValue();
    // auto idx = 0;
    // for (const auto &name : overrides) {
    //     if (idx < static_cast<int>(replay.players.size())) {
    //         replay.players.at(hlt::Player::id_type{idx}).name = name;
    //     }
    //     idx++;
    // }

    // // JSON results info, used by backend
    // nlohmann::json results;
    // results["error_logs"] = nlohmann::json::object();
    // results["terminated"] = nlohmann::json::object();

    // Output replay file for visualizer

    // While compilers like G++4.8 report C++11 compatibility, they do not
    // support std::put_time, so we have to use strftime instead.
    const auto time = std::time(nullptr);
    const auto localtime = std::localtime(&time);
    static constexpr size_t MAX_DATE_STRING_LENGTH = 25;
    char time_string[MAX_DATE_STRING_LENGTH];
    std::strftime(time_string, MAX_DATE_STRING_LENGTH, "%Y%m%d-%H%M%S%z", localtime);

    // if (!no_replay_switch.getValue()) {
    //     // Output gamefile. First try the replays folder; if that fails, just use the straight filename.
    //     std::stringstream filename_buf;
    //     filename_buf << "replay-" << std::string(time_string);
    //     filename_buf << "-" << replay.map_generator_seed;
    //     filename_buf << "-" << map.width;
    //     filename_buf << "-" << map.height << ".hlt";
    //     auto filename = filename_buf.str();
    //     std::string output_filename = replay_directory + filename;
    //     //results["replay"] = output_filename;
    //     bool enable_compression = !no_compression_switch.getValue();
    //     try {
    //         replay.output(output_filename, enable_compression);
    //     } catch (std::runtime_error &e) {
    //         Logging::log("Error: could not write replay to directory " + replay_directory + ", falling back on current directory.", Logging::Level::Error);
    //         replay_directory = "./";
    //         output_filename = replay_directory + filename;
    //         replay.output(output_filename, enable_compression);
    //     }
    //     Logging::log("Opening a file at " + output_filename);
    // }

    for (const auto &stats : replay.game_statistics.player_statistics) {
        std::stringstream message;
        message << "Player "
                << to_string(stats.player_id)
                << ", '"
                << replay.players.at(stats.player_id).name
                << "', was rank "
                << std::to_string(stats.rank)
                << " with "
                << std::to_string(stats.turn_productions.back())
                << " halite";
        Logging::log(message.str());
    }

    for (const auto &[player_id, player] : replay.players) {
        std::string error_log = game.logs.str(player_id);
        if (!error_log.empty()) {
            // if (!no_logs_switch.getValue() || player.terminated) {
            //     std::stringstream logname_buf;
            //     logname_buf << "errorlog-" << std::string(time_string)
            //                 << "-" << replay.map_generator_seed
            //                 << "-" << map_width
            //                 << "-" << map_height
            //                 << "-" << player_id
            //                 << ".log";
            //     const auto log_filename = logname_buf.str();
            //     auto log_filepath = replay_directory + log_filename;

            //     std::ofstream log_file;
            //     log_file.open(log_filepath, std::ios_base::out);
            //     if (!log_file.is_open()) {
            //         log_filepath = replay_directory + log_filename;
            //         log_file.open(log_filepath, std::ios_base::out);
            //     }

            //     //results["error_logs"][to_string(player_id)] = log_filepath;
            //     log_file.write(error_log.c_str(), error_log.size());
            //     Logging::log("Player has log output. Writing a log at " + log_filepath,
            //                  Logging::Level::Info, player.id);
            // }
            // else {
            //     Logging::log("Player has log output, but log was suppressed.",
            //                  Logging::Level::Info, player.id);
            // }
            //results["terminated"][to_string(player_id)] = player.terminated;
        }
    }



    //results["map_width"] = map_width;
    //results["map_height"] = map_height;
    //results["map_seed"] = seed;
    std::ostringstream stream;
    stream << type;
    //results["map_generator"] = stream.str();
    //results["final_snapshot"] = game.to_snapshot(map_parameters);
    //results["stats"] = nlohmann::json::object();
    for (const auto &stats : replay.game_statistics.player_statistics) {
        // results["stats"][to_string(stats.player_id)] = {
        //     {"rank", stats.rank},
        //     {"score", stats.turn_productions.back()}
        // };
    }

    // if (json_results_switch.getValue()) {
    //     std::cout << results.dump(JSON_INDENT_LEVEL) << std::endl;
    // }

    return 0;
}
