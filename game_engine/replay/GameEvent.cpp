#include "GameEvent.hpp"

#include "nlohmann/json.hpp"

/** A JSON key and value corresponding to a field. */
#define FIELD_TO_JSON(x) {#x, x}

namespace hlt {

/** The JSON key for game event type. */
constexpr auto JSON_TYPE_KEY = "type";

/**
 * Convert a GameEvent to JSON format.
 * @param[out] json The output JSON.
 * @param cell The gameEvent to convert.
 */
void to_json(nlohmann::json &json, const GameEvent &game_event) { game_event->to_json(json); }

/**
 * Convert spawn event to json format
 * @param[out] json JSON to be filled by spawn event
 */
void SpawnEvent::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, GAME_EVENT_TYPE_NAME},
            FIELD_TO_JSON(location),
            FIELD_TO_JSON(owner_id),
            FIELD_TO_JSON(id),
            FIELD_TO_JSON(energy)};
}

/**
 * Convert capture event to json format
 * @param[out] json JSON to be filled by spawn event
 */
void CaptureEvent::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, GAME_EVENT_TYPE_NAME},
            FIELD_TO_JSON(location),
            FIELD_TO_JSON(old_owner),
            FIELD_TO_JSON(new_owner),
            FIELD_TO_JSON(old_id),
            FIELD_TO_JSON(new_id)};
}

/**
 * Convert death event to json format
 * @param[out] json JSON to be filled with death event
 */
void CollisionEvent::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, GAME_EVENT_TYPE_NAME},
            FIELD_TO_JSON(location),
            FIELD_TO_JSON(ships)};
}

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

/**
 * Convert death event to json format
 * @param[out] json JSON to be filled with death event
 */
void ConstructionEvent::to_json(nlohmann::json &json) const {
    json = {{JSON_TYPE_KEY, GAME_EVENT_TYPE_NAME},
            FIELD_TO_JSON(location),
            FIELD_TO_JSON(owner_id),
            FIELD_TO_JSON(id)};
}


}
