#include "hlt/game.hpp"
#include "hlt/constants.hpp"
#include "hlt/log.hpp"
#include "types.hpp"
#include "batcher.hpp"
#include "model.hpp"

#include <random>
#include <ctime>

#include <torch/torch.h>

using namespace std;
using namespace hlt;


torch::Tensor convertEntityStateToTensor(std::shared_ptr<EntityState> &entityStatePtr) {

    auto entityState = entityStatePtr.get();
    auto gameState = entityState->gameState.get();
    auto playerId = entityState->playerId;
    //Halite Location
    //auto halite_location = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto steps_remaining = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    // //My global info
    auto my_ships = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_ships_halite = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_dropoffs = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto my_score = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    //Enemy global info
    auto enemy_ships = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_ships_halite = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_dropoffs = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto enemy_score = torch::zeros({GAME_HEIGHT, GAME_WIDTH});

    //Ship specific information
    auto entity_location = torch::zeros({GAME_HEIGHT, GAME_WIDTH});
    auto entity_energy = torch::zeros({GAME_HEIGHT, GAME_WIDTH});

    entity_location[entityState->entityY][entityState->entityX] = 1;
    entity_energy[entityState->entityY][entityState->entityX] = entityState->halite_on_ship;

    steps_remaining.fill_(gameState->steps_remaining);

    //TODO: Generalize for more players
    if(playerId == 0) {
        my_score.fill_(gameState->scores[0]);
        enemy_score.fill_(gameState->scores[1]);
    } else {
        my_score.fill_(gameState->scores[1]);
        enemy_score.fill_(gameState->scores[0]);
    }

    float haliteLocationArray[GAME_HEIGHT][GAME_WIDTH];

    for(std::size_t y = 0; y < GAME_HEIGHT; y++) {
        for(std::size_t x = 0; x < GAME_WIDTH; x++) {
            auto cell = gameState->position[y][x];
            haliteLocationArray[y][x] = cell.halite_on_ground;

            if(cell.shipId == playerId) {
                my_ships[y][x] = 1;
                my_ships_halite[y][x] = cell.halite_on_ship;
            }
            else if (cell.shipId != -1) {
                enemy_ships[y][x] = 1;
                enemy_ships_halite[y][x] = cell.halite_on_ship;
            }

            if(cell.structureOwner == playerId) {
                my_dropoffs[y][x] = 1;
            }
            else if (cell.structureOwner != -1) {
                enemy_dropoffs[y][x] = 1;
            }
        }
    }

    auto halite_location = torch::from_blob(haliteLocationArray, {GAME_HEIGHT, GAME_WIDTH});
    std::vector<torch::Tensor> frames {halite_location, steps_remaining,
    my_ships, my_ships_halite, my_dropoffs, my_score,
    enemy_ships, enemy_ships_halite, enemy_dropoffs, enemy_score,
    entity_location, entity_energy};

    auto stateTensor = torch::stack(frames);
    return stateTensor;
}

std::shared_ptr<EntityState> parseGameIntoEntityState(std::shared_ptr<GameState> &gameState, long playerId, int entityY, int entityX, float entityEnergy) {
    auto entityStatePtr = std::make_shared<EntityState>();
    auto entityState = entityStatePtr.get();

    entityState->gameState = gameState;
    entityState->entityY = entityY;
    entityState->entityX = entityX;
    entityState->halite_on_ship = (entityEnergy / MAX_HALITE_ON_SHIP) - 0.5;
    entityState->playerId = playerId;

    return entityStatePtr;
}



std::shared_ptr<GameState> parseGameIntoGameState(hlt::Game &game) {

    auto gameStatePtr = std::make_shared<GameState>();
    auto gameState = gameStatePtr.get();

    int no_of_rows = GAME_HEIGHT;
    int no_of_cols = GAME_WIDTH;

    int numRows = game.game_map.get()->cells.size();
    int totalSteps = 0;
    if (numRows == 64) {
        totalSteps = 501;
    }
    else if (numRows == 56) {
        totalSteps = 476;
    }
    else if (numRows == 48) {
        totalSteps = 451;
    }
    else if (numRows == 40) {
        totalSteps = 426;
    }
    else {
        totalSteps = 401;
    }

   int cellY = 0;
   for(auto row : game.game_map.get()->cells) {
        for (auto cell: row) {
            auto x  = cell.position.x;
            auto y = cell.position.y;

            auto scaled_halite = (cell.halite / MAX_HALITE_ON_MAP) - 0.5;
            gameState->position[y][x].halite_on_ground = scaled_halite;

            if(cell.is_occupied()) {
                //There is a ship here
                auto entity = cell.ship.get();

                gameState->position[y][x].halite_on_ship = (entity->halite / MAX_HALITE_ON_SHIP) - 0.5;
                gameState->position[y][x].shipId = entity->id;
                gameState->position[y][x].shipOwner = entity->owner;
            }
        }
    }

    for(auto playerPtr : game.players) {

        auto player = playerPtr.get();
        //auto player = playerPair.second;
        auto spawn = player->shipyard.get();

        //We consider spawn/factories to be both dropoffs and spawns
        gameState->position[spawn->position.y][spawn->position.x].dropOffPresent = true;
        gameState->position[spawn->position.y][spawn->position.x].spawnPresent = true;
        gameState->position[spawn->position.y][spawn->position.x].structureOwner = player->id;

        for(auto dropoffPair : player->dropoffs) {
            auto dropoff = dropoffPair.second.get();
            gameState->position[dropoff->position.y][dropoff->position.x].dropOffPresent = true;
            gameState->position[dropoff->position.y][dropoff->position.x].structureOwner = player->id;
        }

        // Player score
        auto score = player->halite;
        auto floatScore = (float)score;
        gameState->scores[player->id] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
    }

    //Steps remaining
    auto steps_remaining_value = totalSteps - game.turn_number + 1;
    auto scaled_steps = (steps_remaining_value / totalSteps) - 0.5; //We normalize between [-0.5, 0.5]
    gameState->steps_remaining = scaled_steps;

    return gameStatePtr;
}


int main(int argc, char* argv[]) {

    unsigned int rng_seed;
    if (argc > 1) {
        rng_seed = static_cast<unsigned int>(stoul(argv[1]));
    } else {
        rng_seed = static_cast<unsigned int>(time(nullptr));
    }
    mt19937 rng(rng_seed);

    Game game;

    ActorCriticNetwork myModel(/*training=*/false);
    myModel.to(torch::kCPU);
    torch::load(myModel.conv1, "0conv1.pt");
    torch::load(myModel.conv2, "0conv2.pt");
    torch::load(myModel.conv3, "0conv3.pt");
    torch::load(myModel.fc1, "0fc1.pt");
    torch::load(myModel.fc2, "0fc2.pt");
    torch::load(myModel.fc3, "0fc3.pt");    
    torch::load(myModel.fcSpawn, "0fcSpawn.pt");    
    myModel.to(torch::kCUDA);

    // At this point "game" variable is populated with initial map data.
    // This is a good place to do computationally expensive start-up pre-processing.
    // As soon as you call "ready" function below, the 2 second per turn timer will start.
    game.ready("MyCppBot");

    log::log("Successfully created bot! My Player ID is " + to_string(game.my_id) + ". Bot rng seed is " + to_string(rng_seed) + ".");


    int offset = 0;
    for (;;) {
        game.update_frame();
        shared_ptr<Player> me = game.me;
        unique_ptr<GameMap>& game_map = game.game_map;

        vector<Command> command_queue;
        auto gameState = parseGameIntoGameState(game);

        for (const auto& ship_iterator : me->ships) {

            // Parse current ship into frames for our neural network
            shared_ptr<Ship> ship = ship_iterator.second;

            // Parse the map into inputs for our neural network
            log::log("About to parse frames");
            //auto gameState = parseGameIntoGameState(me->id, game, ship->position.y, ship->position.x, ship->halite);
            auto entityState = parseGameIntoEntityState(gameState, me->id, ship->position.y, ship->position.x, ship->halite);
            log::log("Done parsing frames");
            auto state = convertEntityStateToTensor(entityState);
            //Convert input frames into tensor state
            state = state.unsqueeze(0);
            torch::Tensor emptyAction;
            auto result = myModel.forward(state, emptyAction);
            // Convert to the game's interpreation
            auto action = result.action.item<int64_t>();
            log::log("Action: " + std::to_string(action));

            // Send it 
            //std::string unitCommands[6] = {"N","E","S","W","still","construct"};
            if(action == 0) {
                //N
                command_queue.push_back(ship->move(Direction::NORTH));
            }
            else if(action == 1) {
                //E
                command_queue.push_back(ship->move(Direction::EAST));
            }
            else if(action == 2) {
                //S
                command_queue.push_back(ship->move(Direction::SOUTH));
            }
            else if(action == 3) {
                //W
                command_queue.push_back(ship->move(Direction::WEST));
            }
            else if(action == 4) {
                //Still
                command_queue.push_back(ship->stay_still());
            }
            else if(action == 5) {
                //Construct
                command_queue.push_back(ship->make_dropoff());
            }
            else {
                log::log("ERROR: Received bad action from neural network " + std::to_string(action));
            }
        }

        if (me->halite >= constants::SHIP_COST && !game_map->at(me->shipyard)->is_occupied()) {

            long factoryId = -1;

            auto entityState = parseGameIntoEntityState(gameState, me->id, me->shipyard->position.y, me->shipyard->position.x, 0);
            auto state = convertEntityStateToTensor(entityState);
            state = state.unsqueeze(0);
            torch::Tensor emptyAction;
            auto modelOutput = myModel.forward_spawn(state, emptyAction);

            auto actionIndex = modelOutput.action.item<int64_t>();
            if(actionIndex == 0) {
                std::string command = "spawn";
                command_queue.push_back(me->shipyard->spawn());
            }
        }

        if (!game.end_turn(command_queue)) {
            break;
        }
    }

    return 0;
}
