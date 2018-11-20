#include "GameEvent.hpp"

#include "nlohmann/json.hpp"

namespace hlt {

/** The JSON key for game event type. */
constexpr auto JSON_TYPE_KEY = "type";


/** Update statistics after collision */
void CollisionEvent::update_stats(const Store &store, const Map &map,
                                  GameStatistics &stats) {
    (void) map;

    ordered_id_map<Player, int> ships_involved;
    for (const auto &ship_id : ships) {
        const auto &entity = store.get_entity(ship_id);
        stats.player_statistics.at(entity.owner.value).all_collisions++;
        ships_involved[entity.owner]++;
    }
    for (const auto &[player_id, num_ships] : ships_involved) {
        // Increment self-collision to account for uncounted ship
        if (num_ships > 1) {
            stats.player_statistics.at(player_id.value).self_collisions += num_ships;
        }
    }
}


}
