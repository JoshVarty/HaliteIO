#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include "types.hpp"


struct ActorCriticNetwork : torch::nn::Module {
public:

    ActorCriticNetwork(bool training)
    :   conv1(torch::nn::Conv2dOptions(NUMBER_OF_FRAMES, 32, /*kernel_size=*/7)),
        conv2(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/3)),
        conv3(torch::nn::Conv2dOptions(64, 64, /*kernel_size=*/3)),
        fc1(64 * (GAME_HEIGHT - 10) * (GAME_WIDTH - 10), 512),
        fc2(512, 6),               //Actor head - Ship
        fc3(512, 1),               //Critic head
        fcSpawn(512, 2),           //Actor head - Spawn
        device(torch::Device(torch::kCUDA))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fcSpawn", fcSpawn);

        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            device_type = torch::kCUDA;
        } else {
            device_type = torch::kCPU;
        }

        device = torch::Device(device_type);
        this->to(device);

        if(training) {
            //Print out network information at beginning of run
            std::cout << "Conv1: (";
            std::cout << conv1.get()->weight.size(0) << ", ";
            std::cout << conv1.get()->weight.size(1) << ", ";
            std::cout << conv1.get()->weight.size(2) << ", ";
            std::cout << conv1.get()->weight.size(3) << ")" << std::endl;

            std::cout << "Conv2: (";
            std::cout << conv2.get()->weight.size(0) << ", ";
            std::cout << conv2.get()->weight.size(1) << ", ";
            std::cout << conv2.get()->weight.size(2) << ", ";
            std::cout << conv2.get()->weight.size(3) << ")" << std::endl;

            std::cout << "Conv3: (";
            std::cout << conv3.get()->weight.size(0) << ", ";
            std::cout << conv3.get()->weight.size(1) << ", ";
            std::cout << conv3.get()->weight.size(2) << ", ";
            std::cout << conv3.get()->weight.size(3) << ")" << std::endl;

            std::cout << "fc1: (";
            std::cout << fc1.get()->weight.size(0) << ", ";
            std::cout << fc1.get()->weight.size(1) << ")" << std::endl;

            std::cout << "fc2: (";
            std::cout << fc2.get()->weight.size(0) << ", ";
            std::cout << fc2.get()->weight.size(1) << ")" << std::endl;

            std::cout << "fc3: (";
            std::cout << fc3.get()->weight.size(0) << ", ";
            std::cout << fc3.get()->weight.size(1) << ")" << std::endl;

            std::cout << "fcSpawn: (";
            std::cout << fcSpawn.get()->weight.size(0) << ", ";
            std::cout << fcSpawn.get()->weight.size(1) << ")" << std::endl;
        }
    }

    ModelOutput forward_spawn(torch::Tensor x, torch::Tensor selected_action) {
        x = x.to(this->device);
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = x.view({-1, 64 * (GAME_HEIGHT - 10) * (GAME_WIDTH - 10)});
        x = torch::relu(fc1->forward(x));

        auto a = fcSpawn->forward(x);
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
        auto p_log_p = action_probabilities * action_probabilities.log();
        auto entropy = -p_log_p.sum(-1).unsqueeze(-1);
        auto value = fc3->forward(x);
        //Return action, log_prob, value, entropy
        ModelOutput output {selected_action, log_prob, value, entropy};
        return output;
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
        auto p_log_p = action_probabilities * action_probabilities.log();
        auto entropy = -p_log_p.sum(-1).unsqueeze(-1);
        auto value = fc3->forward(x);
        //Return action, log_prob, value, entropy
        ModelOutput output {selected_action, log_prob, value, entropy};
        return output;
  }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fcSpawn;
    
    torch::Device device;
};

#endif
