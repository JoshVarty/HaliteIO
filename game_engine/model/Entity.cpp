#include "Entity.hpp"

namespace hlt {

/**
 * Write an Entity to bot serial format.
 * @param ostream The output stream.
 * @param entity The entity to write.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &ostream, const Entity &entity) {
    // Output the entity ID, then energy.
    return ostream << entity.id << " " << entity.energy;
}

}
