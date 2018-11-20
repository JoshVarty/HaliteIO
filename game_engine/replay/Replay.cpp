#include "Replay.hpp"
#include "Logging.hpp"
namespace hlt {

/**
 * Given the game store, reformat and store entity state at start of turn in replay
 * param store The game store at the start of the turn
 */
void Turn::add_entities(Store &store) {
    // Initialize each player to have no entities
    for (const auto &[player_id, _player] : store.players) {
        entities[player_id] = {};
    }
    for (const auto &[entity_id, entity] : store.entities) {
        const auto location = store.get_player(entity.owner).get_entity_location(entity.id);
        const EntityInfo entity_info = {location, entity};
        entities[entity.owner].insert( {{entity.id, entity_info}} );
    }
}

/**
 * Add cells changed on this turn to the replay file
 * @param map The game map (to access cell energy)
 * @param cells The locations of changed cells
 */
void Turn::add_cells(Map &map, std::unordered_set<Location> changed_cells){
    for (const auto location : changed_cells) {
        const auto cell = map.at(location);
        this->cells.emplace_back(location, cell);
    }
}

/**
 * Given the game store, add all state from end of turn in replay
 * param store The game store at the end of the turn
 */
void Turn::add_end_state(Store &store) {
    for (const auto &[player_id, player] : store.players) {
        energy.insert({player_id, player.energy});
        deposited.insert({player_id, player.total_energy_deposited});
    }
}

/**
 * Output replay into file. Replay will be in json format and may be compressed
 *
 * @param filename File to put replay into
 * @param enable_compression Switch to decide whether or not to compress replay file
 */
void Replay::output(std::string filename, bool enable_compression) {
   
}

}
