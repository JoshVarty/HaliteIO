#include "hlt/game.hpp"
#include "hlt/constants.hpp"
#include "hlt/log.hpp"

#include <random>
#include <ctime>

#include <torch/torch.h>

using namespace std;
using namespace hlt;

struct StepResult {
    double meanScore;
    double meanSteps;
};


const float MAX_HALITE_ON_MAP = 1000;           //The maximum natural drop of halite on the map
const float MAX_HALITE_ON_SHIP = 1000;          //The maximum halite a ship can hold
const float MAX_SCORE_APPROXIMATE = 50000;      //A rough estimate of a "Max" score that we'll use for scaling our player's scores

const int NUMBER_OF_FRAMES = 12;                //The number of NxN input frames to our neural network
const int GAME_WIDTH = 32;                //The number of NxN input frames to our neural network
const int GAME_HEIGHT = 32;                //The number of NxN input frames to our neural network

struct Frame {
public:
    float state[NUMBER_OF_FRAMES][GAME_HEIGHT][GAME_WIDTH] = {};

    // void debug_print() {
    //     for(int i = 0; i < NUMBER_OF_FRAMES; i++){
    //         std::cout << std::endl << std::endl << "FRAME: " << i << std::endl;

    //         for(int j = 0; j < GAME_HEIGHT; j++) {
    //             std::cout << std::endl << j << ": \t";
    //             for(int k = 0; k < GAME_WIDTH; k++){
    //                 std::cout << state[i][j][k] << " ";
    //             }
    //         }
    //     }
    // }
};


struct ModelOutput {
    at::Tensor action;
    at::Tensor log_prob;
    at::Tensor value;
};

struct RolloutItem {
    Frame state;
    long action;
    float value;
    float log_prob;
    float reward;
    int done;
    long playerId;
};

struct CompleteRolloutResult {
    std::vector<RolloutItem> rollouts;
    std::vector<long> scores;
    std::vector<long> gameSteps;
};

struct ProcessedRolloutItem {
    Frame state;
    long action;
    float log_prob;
    float returns;
    float advantage;
};

class Batcher {
public:
int batchSize;
int numEntries;
std::vector<ProcessedRolloutItem> data;
std::vector<ProcessedRolloutItem>::iterator batchStart;
std::vector<ProcessedRolloutItem>::iterator batchEnd;

    Batcher(int batchSize, std::vector<ProcessedRolloutItem> data) {
        this->batchSize = batchSize;
        this->data = data;
        this->numEntries = data.size();
        this->reset();
    }

    void reset() {
        this->batchStart = data.begin();
        this->batchEnd = data.begin();
        advance(batchEnd, batchSize);
    }

    bool end() {
        return this->batchStart == this->data.end();
    }

    std::vector<ProcessedRolloutItem> next_batch(){
        std::vector<ProcessedRolloutItem> batch;
        batch.insert(batch.end(), batchStart, batchEnd);

        batchStart = batchEnd; 

        //Advance batchEnd to either the end of the next batch or
        //to the end of the data, whichever comes first
        for(int i = 0; i < batchSize; i++){
            batchEnd = batchEnd + 1;
            if(batchEnd == this->data.end()) {
                break;
            }
        }

        return batch;
    }

    void shuffle() {
        std::random_shuffle(this->data.begin(), this->data.end());
        this->reset();
    }
};


struct ActorCriticNetwork : torch::nn::Module {
public:

    torch::Device device;

    ActorCriticNetwork()
    :   conv1(torch::nn::Conv2dOptions(NUMBER_OF_FRAMES, 32, /*kernel_size=*/7)),
        conv2(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/3)),
        conv3(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3)),
         fc1(64 * (GAME_HEIGHT - 10) * (GAME_WIDTH - 10), 512),
         fc2(512, 6),           //Actor head
         fc3(512, 1),           //Critic head
         device(torch::Device(torch::kCUDA))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);

        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            device_type = torch::kCUDA;
        } else {
            device_type = torch::kCPU;
        }

        device = torch::Device(device_type);
        this->to(device);
    }

    ModelOutput forward(torch::Tensor x, torch::Tensor selected_action) {
        x = x.to(this->device);
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = x.view({-1, 64 * (GAME_HEIGHT - 10) * (GAME_WIDTH - 10)});
        x = torch::relu(fc1->forward(x));

        auto a = fc2->forward(x);
        auto action_probabilities = torch::softmax(a, /*dim=*/1);

        if(selected_action.numel() == 0) {
            //See:  https://github.com/pytorch/pytorch/blob/f79fb58744ba70970de652e46ea039b03e9ce9ff/torch/distributions/categorical.py#L110
            //      https://pytorch.org/cppdocs/api/function_namespaceat_1ac675eda9cae4819bc9311097af498b67.html?highlight=multinomial
            selected_action = action_probabilities.multinomial(1);
            // std::cout << a << std::endl;
            // std::cout << action_probabilities << std::endl;
        }
        else {
            selected_action = selected_action.to(device);
        }

        auto log_prob = action_probabilities.gather(1, selected_action).log();
        auto value = fc3->forward(x);
        //Return action, log_prob, value
        ModelOutput output {selected_action, log_prob, value};
        return output;
  }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
};


Frame parseGridIntoSlices(long playerId, hlt::Game &game) {

    int no_of_rows = GAME_HEIGHT;
    int no_of_cols = GAME_WIDTH;
    int offset = 0;

    //int numRows = game.map.grid.size();
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

    //Board info
    Frame myFrame;
    auto frameData = myFrame.state;
    auto halite_locations = frameData[0];   // Range from [-0.5, ~0.5]
    auto steps_remaining = frameData[1];    // Range from [-0.5, 0.5]
    //My global info
    auto my_ships = frameData[2];           // Range from [0,1]
    auto my_ships_halite = frameData[3];    // Range from [-0.5, 0.5]
    auto my_dropoffs = frameData[4];        // Range from [0, 1]
    auto my_score = frameData[5];           // Range from [-0.5, 0.5]
    //Enemy global info
    auto enemy_ships = frameData[6];        // Range from [0,1]
    auto enemy_ships_halite = frameData[7]; // Range from [-0.5, 0.5]
    auto enemy_dropoffs = frameData[8];     // Range from [0,1]
    auto enemy_score = frameData[9];        // Range from [-0.5, 0.5]

    for(auto row : game.game_map.get()->cells) {
        for (auto cell: row) {
            auto x = offset + cell.position.x;
            auto y = offset + cell.position.y;

            auto scaled_halite = (cell.halite / MAX_HALITE_ON_MAP) - 0.5;
            halite_locations[y][x] = scaled_halite;

            if(cell.ship.get() != nullptr) {
                //There is a ship here
                //auto entity = game.store.get_entity(cell.entity);
                auto entity = cell.ship.get();

                if(entity->owner == playerId) {
                    my_ships_halite[y][x] = (entity->halite / MAX_HALITE_ON_SHIP) - 0.5;
                    my_ships[y][x] = 1;
                }
                else {
                    enemy_ships_halite[y][x] = (entity->halite / MAX_HALITE_ON_SHIP) - 0.5;
                    enemy_ships[y][x] = 2;
                }
            }
        }
    }

    for(auto playerPtr : game.players) {

        auto player = playerPtr.get();
        //auto player = playerPair.second;
        auto spawn = player->shipyard.get();
        auto x = offset + spawn->position.x;
        auto y = offset + spawn->position.y;

        if(player->id == playerId) {
            //We mark our spawn as a 'dropoff' because it can also be used as one
            my_dropoffs[y][x] = 1;
        }
        else {
            //We mark the enemy spawn as a 'dropoff' because it can also be used as one
            enemy_dropoffs[y][x] = 2;
        }

        for(auto dropoff : player->dropoffs) {
            if(player->id == playerId) {
                my_dropoffs[y][x] = 1;
            }
            else {
                //We mark the enemy spawn as a 'dropoff' because it can also be used as one
                enemy_dropoffs[y][x] = 2;
            }
        }

        // Player score
        auto score = player->halite;
        auto floatScore = (float)score;
        if(player->id == playerId) {
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    my_score[i][j] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
                }
            }
        }
        else {
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    enemy_score[i][j] = (floatScore / MAX_SCORE_APPROXIMATE) - 0.5;
                }
            }
        }

        //Steps remaining
        auto steps_remaining_value = totalSteps - game.turn_number + 1;
        if(player->id == playerId) {
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++){
                    steps_remaining[i][j] = (steps_remaining_value / totalSteps) - 0.5; //We normalize between [-0.5, 0.5]
                }
            }
        }
    }

    return myFrame;
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

    ActorCriticNetwork myModel;
    myModel.to(torch::kCPU);
    torch::load(myModel.conv1, "9conv1.pt");
    torch::load(myModel.conv2, "9conv2.pt");
    torch::load(myModel.conv3, "9conv3.pt");
    torch::load(myModel.fc1, "9fc1.pt");
    torch::load(myModel.fc2, "9fc2.pt");
    torch::load(myModel.fc3, "9fc3.pt");    
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

        for (const auto& ship_iterator : me->ships) {
            // shared_ptr<Ship> ship = ship_iterator.second;
            // if (game_map->at(ship)->halite < constants::MAX_HALITE / 10 || ship->is_full()) {
            //     Direction random_direction = ALL_CARDINALS[rng() % 4];
            //     command_queue.push_back(ship->move(random_direction));
            // } else {
            //     command_queue.push_back(ship->stay_still());
            // }

            // Parse the map into inputs for our neural network
            log::log("About to parse frames");
            auto frames = parseGridIntoSlices(me->id, game);
            log::log("Done parsing frames");
            // Parse current ship into frames for our neural network
            shared_ptr<Ship> ship = ship_iterator.second;

            //Zero out all cells except for where our current unit is
            auto entityLocationFrame = frames.state[10];
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++) {
                    entityLocationFrame[i][j] = 0;
                }
            }
            entityLocationFrame[offset + ship->position.y][offset + ship->position.x] = 1;

            //Set entire frame to the score of the current unit
            auto entityEnergyFrame = frames.state[11];
            float energy = ship->halite;
            for(int i = 0; i < GAME_HEIGHT; i++) {
                for(int j = 0; j < GAME_WIDTH; j++) {
                    entityEnergyFrame[i][j] = (energy / MAX_HALITE_ON_SHIP) - 0.5;
                }
            }

            //Conver input frames into tensor state
            auto state = torch::from_blob(frames.state, {NUMBER_OF_FRAMES, GAME_HEIGHT, GAME_WIDTH});
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

        if (game.turn_number <= 200 && me->halite >= constants::SHIP_COST && !game_map->at(me->shipyard)->is_occupied()) {
            command_queue.push_back(me->shipyard->spawn());
        }

        if (!game.end_turn(command_queue)) {
            break;
        }
    }

    return 0;
}
