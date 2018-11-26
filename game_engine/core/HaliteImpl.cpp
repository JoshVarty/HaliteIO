#include <future>
#include <set>

#include "HaliteImpl.hpp"

namespace hlt {

/**
 * Initialize the game.
 * @param player_commands The list of player commands.
 */
void HaliteImpl::initialize_game(int n_players) {
    // Update max turn # by map size (300 @ 32x32 to 500 at 80x80)
    auto &mut_constants = Constants::get_mut();
    auto turns = mut_constants.MIN_TURNS;
    const unsigned long max_dimension = std::max(game.map.width, game.map.height);
    if (max_dimension > mut_constants.MIN_TURN_THRESHOLD) {
        turns += static_cast<unsigned long>(((max_dimension - mut_constants.MIN_TURN_THRESHOLD) / static_cast<double>(mut_constants.MAX_TURN_THRESHOLD - mut_constants.MIN_TURN_THRESHOLD)) * (mut_constants.MAX_TURNS - mut_constants.MIN_TURNS));
    }
    mut_constants.MAX_TURNS = turns;

    const auto &constants = Constants::get();
    auto &players = game.store.players;
    //assert(game.map.factories.size() >= player_commands.size());

    // Add a 0 frame so we can record beginning-of-game state
    game.replay.full_frames.emplace_back();
    std::unordered_set<Location> changed_cells;

    auto factory_iterator = game.map.factories.begin();

    for (dimension_type row = 0; row < game.map.height; row++) {
        for (dimension_type col = 0; col < game.map.width; col++) {
            game.store.map_total_energy += game.map.at(col, row).energy;
        }
    }

    //for (const auto &command : player_commands) {
    for (int i = 0; i < n_players; i++) {
        auto &factory = *factory_iterator++;
        auto player = game.store.player_factory.make(factory);
        player.energy = constants.INITIAL_ENERGY;
        game.game_statistics.player_statistics.emplace_back(player.id, game.rng());
        players.emplace(player.id, player);
    }
    game.replay.game_statistics = game.game_statistics;

    for (auto &[player_id, player] : game.store.players) {
        // Zero the energy on factory and mark as owned.
        auto &factory = game.map.at(player.factory);
        game.store.map_total_energy -= factory.energy;
        factory.energy = 0;
        factory.owner = player_id;
        // Prepare the log.
        //game.logs.add(player_id);
    }
    game.replay.full_frames.back().add_cells(game.map, changed_cells);
    update_player_stats();
    game.replay.full_frames.back().add_end_state(game.store);
}

/** Run the game. */
void HaliteImpl::run_game() {
    //removed
}

void HaliteImpl::parse(AgentCommand agentCommand, std::unique_ptr<Command> &command) {
    // Read one character corresponding to the type, and dispatch the remainder based on its value.
    auto entityId = agentCommand.first;
    auto agentCommandText = agentCommand.second;

    char command_type;
    if (agentCommandText == "N" || agentCommandText == "E" || agentCommandText == "S" || agentCommandText == "W" || agentCommandText == "still") {
        Entity::id_type entity(agentCommand.first);
        Direction direction;

        if (agentCommandText == "N"){
            direction = Direction::North;
        }
        else if (agentCommandText == "E"){
            direction = Direction::East;
        }
        else if (agentCommandText == "S"){
            direction = Direction::South;
        }
        else if (agentCommandText == "W"){
            direction = Direction::West;
        }
        else if (agentCommandText == "still"){
            direction = Direction::Still;
        }
        //istream >> entity >> direction;
        command = std::make_unique<MoveCommand>(entity, direction);
    }
    else if (agentCommandText == "spawn") {
        command = std::make_unique<SpawnCommand>();
    }
    else if (agentCommandText == "construct") {
        Entity::id_type entity(agentCommand.first);
        command = std::make_unique<ConstructCommand>(entity);
    }
    else {
        std::cout << "You didn't set command! What's wrong with you?" << std::endl;
        exit(1);
    }
}

/** Retrieve and process commands, and update the game state for the current turn. */
void HaliteImpl::process_turn(std::map<long, std::vector<AgentCommand>> rawCommands) {
    // Retrieve all commands
    using Commands = std::vector<std::unique_ptr<Command>>;
    ordered_id_map<Player, Commands> commands{};

    auto &players = game.store.players;

    for(uint i = 0; i < rawCommands.size(); i++) {
        auto playerRawCommands = rawCommands[i];

        for(auto playerPair: players){
            //Find the player who owns these commands. This is ugly I know.
            auto playerId = playerPair.first.value;
            auto player = playerPair.second;
            if(playerId == i) {

                Commands currentCommands;

                for (auto rawCommand : playerRawCommands) {

                    std::unique_ptr<Command> command;
                    //Convert from our command system into the game's command system
                    parse(rawCommand, command);
                    currentCommands.push_back(std::move(command));
                }

                commands[playerPair.first] = std::move(currentCommands);
            }
        }
    }


    // Process valid player commands, removing players if they submit invalid ones.
    std::unordered_set<Entity::id_type> changed_entities;
    while (!commands.empty()) {
        changed_entities.clear();
        game.store.changed_cells.clear();

        CommandTransaction transaction{game.store, game.map};
        std::unordered_set<Player::id_type> offenders;
        transaction.on_event([&frames = game.replay.full_frames, this](GameEvent event) {
            event->update_stats(game.store, game.map, game.game_statistics);
            // Create new game event for replay file.
            frames.back().events.push_back(std::move(event));
        });
        transaction.on_error([&offenders, &commands, this](CommandError error) {
            this->handle_error(offenders, commands, std::move(error));
        });
        transaction.on_cell_update([&changed_cells = game.store.changed_cells](Location cell) {
            changed_cells.emplace(cell);
        });
        transaction.on_entity_update([&changed_entities](Entity::id_type entity) {
            changed_entities.emplace(entity);
        });

        for (const auto &[player_id, command_list] : commands) {
            auto &player = game.store.players.find(player_id)->second;
            for (const auto &command : command_list) {
                command->add_to_transaction(player, transaction);
            }
        }
        if (transaction.check()) {
            // All commands are successful.
            transaction.commit();
            if (Constants::get().STRICT_ERRORS) {
                if (!offenders.empty()) {
                    std::cout << "Command processing failed for players: ";
                    for (auto iterator = offenders.begin(); iterator != offenders.end(); iterator++) {
                        std::cout << *iterator;
                        if (std::next(iterator) != offenders.end()) {
                            std::cout << ", ";
                        }
                    }
                    std::cout << ", aborting due to strict error check";
                    //Logging::log(stream.str(), Logging::Level::Error);
                    game.turn_number = Constants::get().MAX_TURNS;
                    exit(1);
                    return;
                }
            } else {
                assert(offenders.empty());
            }
            // Add player commands to replay and note players still alive
            game.replay.full_frames.back().moves = std::move(commands);
            break;
        } else {
            for (auto player : offenders) {
                kill_player(player);
                commands.erase(player);
            }
        }
    }

    // Resolve ship mining
    const auto max_energy = Constants::get().MAX_ENERGY;
    const auto ships_threshold = Constants::get().SHIPS_ABOVE_FOR_CAPTURE;
    const auto bonus_multiplier = Constants::get().INSPIRED_BONUS_MULTIPLIER;
    for (auto &[entity_id, entity] : game.store.entities) {
        if (changed_entities.find(entity_id) == changed_entities.end()
            && entity.energy < max_energy) {
            // Allow this entity to extract
            const auto location = game.store.get_player(entity.owner).get_entity_location(entity_id);
            auto &cell = game.map.at(location);

            const auto ratio = entity.is_inspired ?
                Constants::get().INSPIRED_EXTRACT_RATIO :
                Constants::get().EXTRACT_RATIO;
            energy_type extracted = static_cast<energy_type>(
                std::ceil(static_cast<double>(cell.energy) / ratio));
            energy_type gained = extracted;

            // If energy is small, give it all to the entity.
            if (extracted == 0 && cell.energy > 0) {
                extracted = gained = cell.energy;
            }

            // Don't take more than the entity can hold.
            if (extracted + entity.energy > max_energy) {
                extracted = max_energy - entity.energy;
            }

            // Apply bonus for inspired entities
            if (entity.is_inspired && bonus_multiplier > 0) {
                gained += bonus_multiplier * gained;
            }

            // Do not allow entity to exceed capacity.
            if (max_energy - entity.energy < gained) {
                gained = max_energy - entity.energy;
            }
            auto &player_stats = game.game_statistics.player_statistics.at(entity.owner.value);
            player_stats.total_mined += extracted;
            player_stats.total_bonus += gained > extracted ? gained - extracted : 0;
            if (entity.was_captured) {
                player_stats.total_mined_from_captured += gained;
            }
            entity.energy += gained;
            cell.energy -= extracted;
            game.store.map_total_energy -= extracted;
            game.store.changed_cells.emplace(location);
        }
    }

    // Resolve ship capture
    if (Constants::get().CAPTURE_ENABLED) {
        const auto capture_radius = Constants::get().CAPTURE_RADIUS;
        std::unordered_map<Location, Player::id_type> entity_switches;
        for (const auto &[player_id, player] : game.store.players) {
            for (const auto &[entity_id, location] : player.entities) {
                id_map<Player, unsigned long> ships_in_radius;
                for(const auto &[pid, _] : game.store.players) ships_in_radius[pid] = 0;

                std::unordered_set<Location> visited_locs{};
                std::unordered_set<Location> to_visit_locs{location};
                while(!to_visit_locs.empty()) {
                    const Location cur = *to_visit_locs.begin();
                    visited_locs.emplace(cur);
                    to_visit_locs.erase(to_visit_locs.begin());

                    Cell cur_cell = game.map.at(cur);
                    if(cur_cell.entity != Entity::None) {
                        ships_in_radius[game.store.get_entity(cur_cell.entity).owner]++;
                    }

                    for(const Location &neighbor : game.map.get_neighbors(cur)) {
                        if((visited_locs.find(neighbor) == visited_locs.end())
                           && (to_visit_locs.find(neighbor) == to_visit_locs.end())) {
                            if(game.map.distance(location, neighbor) <= capture_radius) {
                                to_visit_locs.emplace(neighbor);
                            } else {
                                visited_locs.emplace(neighbor);
                            }
                        }
                    }
                }

                unsigned long max_val = 0;
                Player::id_type max_id = Player::None;
                for(const auto &[pid, val] : ships_in_radius) {
                    if(pid != player_id && val > max_val) {
                        max_val = val;
                        max_id = pid;
                    }
                }
                if(ships_in_radius[player_id]+ships_threshold <= max_val) {
                    entity_switches[location] = max_id;
                }
            }
        }

        // Flip the ships that have been captured
        for(const auto &[location, new_player_id] : entity_switches) {
            auto &cell = game.map.at(location);
            const auto entity = game.store.get_entity(cell.entity);

            game.game_statistics.player_statistics.at(entity.owner.value).ships_given++;
            game.game_statistics.player_statistics.at(new_player_id.value).ships_captured++;

            game.store.delete_entity(entity.id);
            auto &entities = game.store.get_player(entity.owner).entities;
            entities.erase(entities.find(entity.id));

            auto new_entity = game.store.new_entity(entity.energy, new_player_id);
            // XXX new_entity seems to be a copy not a real reference
            cell.entity = new_entity.id;
            game.store.get_entity(new_entity.id).was_captured = true;
            game.store.get_player(new_player_id).add_entity(new_entity.id, location);

            game.replay.full_frames.back().events.push_back(
                                                            std::make_unique<CaptureEvent>(location, entity.owner, entity.id,
                                                                                           new_player_id, new_entity.id));
        }
    }


    game.replay.full_frames.back().add_cells(game.map, game.store.changed_cells);
    update_player_stats();
}

void HaliteImpl::update_inspiration() {
    if (!Constants::get().INSPIRATION_ENABLED) {
        return;
    }

    const auto inspiration_radius = Constants::get().INSPIRATION_RADIUS;
    const auto ships_threshold = Constants::get().INSPIRATION_SHIP_COUNT;

    // Check every ship of every player
    for (const auto &[player_id, player] : game.store.players) {
        for (const auto &[entity_id, location] : player.entities) {
            // map from player ID to # of ships within the inspiration
            // radius of the current ship
            id_map<Player, long> ships_in_radius;
            for (const auto &[pid, _] : game.store.players) {
                ships_in_radius[pid] = 0;
            }

            // Explore locations around this ship
            for (auto dx = -inspiration_radius; dx <= inspiration_radius; dx++) {
                for (auto dy = -inspiration_radius; dy <= inspiration_radius; dy++) {
                    const auto cur = Location{
                        (((location.x + dx) % game.map.width) + game.map.width) % game.map.width,
                        (((location.y + dy) % game.map.height) + game.map.height) % game.map.height
                    };

                    const auto &cur_cell = game.map.at(cur);
                    if (cur_cell.entity == Entity::None ||
                        game.map.distance(location, cur) > inspiration_radius) {
                        continue;
                    }
                    const auto &other_entity = game.store.get_entity(cur_cell.entity);
                    ships_in_radius[other_entity.owner]++;
                }
            }

            // Total up ships of other player
            unsigned long opponent_entities = 0;
            for (const auto &[pid, count] : ships_in_radius) {
                if (pid != player_id) {
                    opponent_entities += count;
                }
            }

            // Mark ship as inspired or not
            auto &entity = game.store.get_entity(entity_id);
            entity.is_inspired = opponent_entities >= ships_threshold;
        }
    }
}

/**
 * Determine whether a player can still play in the future
 *
 * @param player Player to check
 * @return True if the player can play on the next turn
 */
bool HaliteImpl::player_can_play(const Player &player) const {
    return !player.entities.empty() || player.energy >= Constants::get().NEW_ENTITY_ENERGY_COST;
}
/**
 * Determine whether the game has ended.
 * @return True if the game has ended.
 */
bool HaliteImpl::game_ended() const {
    if (game.store.map_total_energy == 0) {
        auto all_entities = game.store.all_entities();
        return std::all_of(all_entities.begin(),
                           all_entities.end(),
                           [](const auto& entity) {
                               return entity.second.energy == 0;
                           });
    }
    long num_alive_players = 0;
    for (auto &&[_, player] : game.store.players) {
        bool can_play = player_can_play(player);
        if (player.can_play && !can_play) {
            //Logging::log("player has insufficient resources to continue", Logging::Level::Info, player.id);
            std::cout << "player has insufficient resources to continue: " << player.id.value << std::endl;
            player.can_play = false;
        }
        if (!player.terminated && can_play) {
            num_alive_players++;
        }
        if (num_alive_players > 1) {
            return false;
        }
    }
    // If there is only one player in the game, then let them keep playing.
    return !(game.store.players.size() == 1 && num_alive_players == 1);
}

/**
 * Update all players' statistics after a single turn.
 */
void HaliteImpl::update_player_stats() {
    for (PlayerStatistics &player_stats : game.game_statistics.player_statistics) {
        // Player with sprites is still alive, so mark as alive on this turn and add production gained
        const auto &player_id = player_stats.player_id;
        const Player &player = game.store.get_player(player_id);
        if (!player.terminated && player.can_play) {
            if (player_can_play(player)) { // Player may have died during this turn, in which case do not update final turn
                player_stats.last_turn_alive = game.turn_number;
            }
            player_stats.turn_productions.push_back(player.energy);
            player_stats.turn_deposited.push_back(player.total_energy_deposited);
            player_stats.number_dropoffs = player.dropoffs.size();
            for (const auto &[_entity_id, location] : player.entities) {
                const dimension_type entity_distance = game.map.distance(location, player.factory);
                if (entity_distance > player_stats.max_entity_distance)
                    player_stats.max_entity_distance = entity_distance;
                player_stats.total_distance += entity_distance;
                player_stats.total_entity_lifespan++;
                if (possible_interaction(player_id, location)) {
                    player_stats.interaction_opportunities++;
                }
            }

            player_stats.halite_per_dropoff[player.factory] = player.factory_energy_deposited;
            player_stats.total_production = player.total_energy_deposited;
            for (const auto &dropoff : player.dropoffs) {
                player_stats.halite_per_dropoff[dropoff.location] = dropoff.deposited_halite;
            }
        } else {
            player_stats.turn_productions.push_back(0);
            player_stats.turn_deposited.push_back(0);
        }
    }
}

/**
 * Determine if entity owned by given player is in range of another player (their entity, dropoff, or factory) and thus may interact
 *
 * param owner_id Id of owner of entity at given location
 * param entity_location Location of entity we are assessing for an interaction opportunity
 * return bool Indicator of whether there players are in close range for an interaction (true) or not (false)
 */
bool HaliteImpl::possible_interaction(const Player::id_type owner_id, const Location entity_location) {
    // Fetch all locations 2 cells away
    std::unordered_set<Location> close_cells{};
    const auto neighbors = game.map.get_neighbors(entity_location);
    close_cells.insert(neighbors.begin(), neighbors.end());
    for (const Location &neighbor : neighbors) {
        const auto cells_once_removed = game.map.get_neighbors(neighbor);
        close_cells.insert(cells_once_removed.begin(), cells_once_removed.end());
    }
    // Interaction possibilty implies a cell has an entity owned by another player or there is a factory or dropoff
    // of another player on the cell. Interactions between entities of a single player are ignored
    for (const Location &cell_location : close_cells) {
        const Cell &cell = game.map.at(cell_location);
        if (cell.entity != Entity::None) {
            if (game.store.get_entity(cell.entity).owner != owner_id) return true;
        }
        if (cell.owner != Player::None && cell.owner != owner_id) return true;
    }
    return false;

}

/**
 * Update players' rankings based on their final turn alive, then break ties with production totals in final turn.
 * Function is intended to be called at end of game, and will in place modify the ranking field of player statistics
 * to rank players from winner (1) to last player.
 */
void HaliteImpl::rank_players() {
    auto &statistics = game.game_statistics.player_statistics;
    std::stable_sort(statistics.begin(), statistics.end());
    // Reverse list to have best players first
    std::reverse(statistics.begin(), statistics.end());

    for (size_t index = 0; index < statistics.size(); ++index) {
        statistics[index].rank = index + 1l;
    }

    // Re-sort by player ID
    std::stable_sort(statistics.begin(), statistics.end(),
                     [](const PlayerStatistics &a, const PlayerStatistics &b) -> bool {
                         return a.player_id.value < b.player_id.value;
                     });
}

void HaliteImpl::kill_player(const Player::id_type &player_id) {
    //Logging::log("Killing player", Logging::Level::Warning, player_id);
    std::cout << "Killing player: " << player_id << std::endl;
    auto &player = game.store.get_player(player_id);
    player.terminated = true;
    //game.networking.kill_player(player);

    auto &entities = player.entities;
    for (auto entity_iterator = entities.begin();
         entity_iterator != entities.end();
         entity_iterator = entities.erase(entity_iterator)) {
        const auto &[entity_id, location] = *entity_iterator;
        auto &cell = game.map.at(location);
        cell.entity = Entity::None;
        game.store.delete_entity(entity_id);
    }
    player.energy = 0;
}

/**
 * Construct HaliteImpl from game interface.
 * @param game The game interface.
 */
HaliteImpl::HaliteImpl(Halite &game) : game(game) {}

/**
 * Handle a player command error.
 * @param offenders The set of players this turn who have caused errors.
 * @param commands The player command mapping.
 * @param error The error caused by the player.
 */
void HaliteImpl::handle_error(std::unordered_set<Player::id_type> &offenders,
                              ordered_id_map<Player, std::vector<std::unique_ptr<Command>>> &commands,
                              CommandError error) {
    const auto message = error->log_message();
    const auto &faulty = error->command();
    const auto player_id = error->player;

    // Log the error information.
    if (error->ignored) {
        //Logging::log(message, Logging::Level::Warning, player_id);
        std::cout << message << " " << player_id << std::endl;
        //game.logs.log(player_id, message, PlayerLog::Level::Warning);
    } else {
        //Logging::log(message, Logging::Level::Error, player_id);
        std::cout << message << " " << player_id << std::endl;
        offenders.emplace(player_id);
        //game.logs.log(player_id, message, PlayerLog::Level::Error);
    }

    // Find the position of a command within a player's command list.
    auto &player_commands = commands[player_id];
    const auto find_position = [&player_commands](const Command &faulty) {
        return std::find_if(player_commands.begin(), player_commands.end(), [&faulty](const auto &command) {
            return std::addressof(*command) == std::addressof(faulty);
        });
    };

    // Given a command position, log the context of that command.
    const auto log_context = [&player_commands, &game = game, &player_id](const auto position) {
        static constexpr long COMMAND_CONTEXT_LINES = 2L;
        const auto distance = static_cast<long>(std::distance(player_commands.begin(), position));
        const auto commands_begin = player_commands.begin() + std::max(0L, distance - COMMAND_CONTEXT_LINES);
        const auto commands_end = player_commands.begin() +
                                  std::min(static_cast<long>(player_commands.size()),
                                           distance + COMMAND_CONTEXT_LINES + 1);
        for (auto iterator = commands_begin; iterator != commands_end; iterator++) {
            auto number = std::distance(player_commands.begin(), iterator);
            auto marker = iterator == position ? ">>> " : "    ";
            //game.logs.log(player_id, marker + std::to_string(number + 1) + "   " + (*iterator)->to_bot_serial());
        }
        //game.logs.log(player_id, "");
    };

    // Log the faulty command.
    auto position = find_position(faulty);
    auto distance = std::distance(player_commands.begin(), position);
    //game.logs.log(player_id, "At command " + std::to_string(distance + 1) + " of " + std::to_string(player_commands.size()) + ":");
    log_context(position);

    // If there is a context, log the context.
    const auto &context = error->context();
    if (!context.empty()) {
        const auto context_message = error->context_message();
        //game.logs.log(player_id, context_message);
        static constexpr size_t MAX_CONTEXT_COMMANDS = 5;
        const auto context_end = context.begin() + std::min(MAX_CONTEXT_COMMANDS, context.size());
        for (auto iterator = context.begin(); iterator != context_end; iterator++) {
            log_context(find_position(*iterator));
        }
        if (context.size() > MAX_CONTEXT_COMMANDS) {
            //game.logs.log(player_id, "(suppressing " + std::to_string(context.size() - MAX_CONTEXT_COMMANDS) + " other commands)\n");
        }
    }
}

}
