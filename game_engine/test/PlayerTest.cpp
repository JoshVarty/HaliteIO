#include <sstream>

#include "nlohmann/json.hpp"

#include "catch.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define protected public
#define private public
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "Player.hpp"

