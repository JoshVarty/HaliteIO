#include "Cell.hpp"

namespace hlt {

/**
 * Write a Cell to bot serial format.
 * @param ostream The output stream.
 * @param cell The cell to write.
 * @return The output stream.
 */
std::ostream &operator<<(std::ostream &ostream, const Cell &cell) {
    return ostream << std::to_string(cell.energy);
}

}
