#include "Statistics.hpp"


namespace hlt {

/**
 * Compare two players to rank them.
 *
 * @param other The statistics of the other player.
 * @return True if this player ranks below (i.e. is worse than) the other.
 */
bool PlayerStatistics::operator<(const PlayerStatistics &other) const {
    if (this->last_turn_alive == other.last_turn_alive) {
        auto turn_to_compare = this->last_turn_alive;
        while (this->turn_productions[turn_to_compare] == other.turn_productions[turn_to_compare]) {
            if (--turn_to_compare < 0) {
                // Players exactly tied on all turns, so randomly choose
                return this->random_id < other.random_id;
            }
        }
        return this->turn_productions[turn_to_compare] < other.turn_productions[turn_to_compare];
    } else {
        return this->last_turn_alive < other.last_turn_alive;
    }
}
}
