#include "BotCommunicationError.hpp"
#include "Location.hpp"

namespace hlt {

/**
 * Convert a char to a Direction.
 * @param direction_char The character.
 * @param[out] direction The converted direction.
 * @return true if the conversion was successful.
 */
bool from_char(const char direction_char, Direction &direction) {
    switch (direction_char) {
    case static_cast<char>(Direction::North):
        direction = Direction::North;
        return true;
    case static_cast<char>(Direction::South):
        direction = Direction::South;
        return true;
    case static_cast<char>(Direction::East):
        direction = Direction::East;
        return true;
    case static_cast<char>(Direction::West):
        direction = Direction::West;
        return true;
    case static_cast<char>(Direction::Still):
        direction = Direction::Still;
        return true;
    default:
        return false;
    }
}

/**
 * Read a Direction from bot serial format.
 * @param istream The input stream.
 * @param[out] direction The direction to read.
 * @return The input stream.
 */
std::istream &operator>>(std::istream &istream, Direction &direction) {
    // Read one character corresponding to the type, and dispatch based on its value.
    char direction_type;
    istream >> direction_type;
    if (!from_char(direction_type, direction)) {
        throw BotCommunicationError(to_string(direction_type));
    }
    return istream;
}


/**
 * Write a Location to bot serial format.
 * @param ostream The output stream.
 * @param location The Location to write.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &ostream, const Location &location) {
    return ostream << location.x << " " << location.y;
}

}
