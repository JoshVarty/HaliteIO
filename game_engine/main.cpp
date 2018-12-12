#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include <iterator>
#include <vector>
#include <algorithm>

//Halite
#include "Constants.hpp"
#include "Generator.hpp"
#include "Halite.hpp"
#include "Replay.hpp"
#include "Enumerated.hpp"

//Torch
#include <torch/torch.h>

//My files
#include "../types.hpp"
#include "../batcher.hpp"
#include "../model.hpp"
#include "agent.hpp"

void ppo(Agent myAgent, uint numEpisodes, int iteration) {
    auto bestMean = -1;
    auto bestNumSteps = -1;
    std::vector<double> allScores;
    std::vector<double> allGameSteps;
    std::vector<double> allLosses;
    std::vector<double> allPolicyLosses;
    std::vector<double> allValueLosses;

    std::deque<double> lastHundredScores;
    std::deque<double> lastHundredSteps;
    std::deque<double> lastHundredLosses;
    std::deque<double> lastHundredValueLosses;
    std::deque<double> lastHundredPolicyLosses;

    for (uint i = 1; i < numEpisodes + 1; i++) {

        StepResult result = myAgent.step();
        allScores.push_back(result.meanScore);
        allGameSteps.push_back(result.meanSteps);
        allLosses.push_back(result.meanLoss);

        allValueLosses.push_back(result.meanValueLoss);
        allPolicyLosses.push_back(result.meanPolicyLoss);
        //Keep track of the last 10 scores
        lastHundredScores.push_back(result.meanScore);
        lastHundredSteps.push_back(result.meanSteps);
        lastHundredLosses.push_back(result.meanLoss);
        lastHundredValueLosses.push_back(result.meanValueLoss);
        lastHundredPolicyLosses.push_back(result.meanPolicyLoss);
        if(lastHundredScores.size() > 50) {
            lastHundredScores.pop_front();
        }
        if(lastHundredSteps.size() > 50) {
            lastHundredSteps.pop_front();
        }
        if(lastHundredLosses.size() > 50) {
            lastHundredLosses.pop_front();
        }
        if(lastHundredValueLosses.size() > 50) {
            lastHundredValueLosses.pop_front();
        }
        if(lastHundredPolicyLosses.size() > 50) {
            lastHundredPolicyLosses.pop_front();
        }

        if (i % 50 == 0) {
            double meanScore = std::accumulate(lastHundredScores.begin(), lastHundredScores.end(), 0.0) / lastHundredScores.size();
            double meanGameSteps = std::accumulate(lastHundredSteps.begin(), lastHundredSteps.end(), 0.0) / lastHundredSteps.size();
            double meanLoss = std::accumulate(lastHundredLosses.begin(), lastHundredLosses.end(), 0.0) / lastHundredLosses.size();
            double meanValueLoss = std::accumulate(lastHundredValueLosses.begin(), lastHundredValueLosses.end(), 0.0) / lastHundredValueLosses.size();
            double meanPolicyLoss = std::accumulate(lastHundredPolicyLosses.begin(), lastHundredPolicyLosses.end(), 0.0) / lastHundredPolicyLosses.size();

            //Every 50 episodes, display the mean reward
            std::cout << "Mean score at step: " << i << ": " << meanScore << std::endl;
            std::cout << "Mean number of gamesteps at step: " << i << ": " << meanGameSteps << std::endl;
            std::cout << "Mean loss at step: " << i << ": " << meanLoss << std::endl;
            std::cout << "Mean value loss at step: " << i << ": " << meanValueLoss << std::endl;
            std::cout << "Mean policy loss at step: " << i << ": " << meanPolicyLoss << std::endl;

            //If our network is improving, save the current weights
            if(meanGameSteps > bestNumSteps || meanScore > bestMean) {
                //TODO: Why do I have to save the weights one-by-one...
                bestMean = meanScore;
                bestNumSteps = meanGameSteps;
                std::cout << "New Best. Saving model..." << std::endl;
                //We have to move the model to CPU before saving
                myAgent.myModel.to(torch::kCPU);
                torch::save(myAgent.myModel.conv1, std::to_string(iteration) + "conv1.pt");
                torch::save(myAgent.myModel.conv2, std::to_string(iteration) + "conv2.pt");
                torch::save(myAgent.myModel.conv3, std::to_string(iteration) + "conv3.pt");
                torch::save(myAgent.myModel.fc1, std::to_string(iteration) + "fc1.pt");
                torch::save(myAgent.myModel.fc2, std::to_string(iteration) + "fc2.pt");
                torch::save(myAgent.myModel.fc3, std::to_string(iteration) + "fc3.pt");
                torch::save(myAgent.myModel.fcSpawn, std::to_string(iteration) + "fcSpawn.pt");
                //Now we move the model back to the GPU
                myAgent.myModel.to(torch::kCUDA);
            }
        }
    }

    std::cout << "AllScores:" << std::endl;
    for(auto score : allScores) {
        std::cout << score << std::endl;
    }

    std::cout << "AllGameSteps:" << std::endl;
    for(auto steps : allGameSteps) {
        std::cout << steps << std::endl;
    }

    std::cout << "AllLosses:" << std::endl;
    for(auto loss : allLosses) {
        std::cout << loss << std::endl;
    }
}

void loadWeights(Agent agent) {
    try {
        agent.myModel.to(torch::kCPU);
        torch::load(agent.myModel.conv1, "0conv1.pt");
        torch::load(agent.myModel.conv2, "0conv2.pt");
        torch::load(agent.myModel.conv3, "0conv3.pt");
        torch::load(agent.myModel.fc1, "0fc1.pt");
        torch::load(agent.myModel.fc2, "0fc2.pt");
        torch::load(agent.myModel.fc3, "0fc3.pt");
        torch::load(agent.myModel.fcSpawn, "0fcSpawn.pt");
    }
    catch (const std::exception& e) {
        std::cout << "Could not load models from disk. Starting from scratch" << std::endl;
        agent.myModel.to(torch::kCUDA);
    }

    agent.myModel.to(torch::kCUDA);
}

void runGridSearch() {
  //Parameters over which we'd like to search
    std::vector<float> discount_rates {0.995};
    std::vector<int> learning_rounds {3, 5, 10};
    std::vector<int> mini_batch_numbers {32, 64, 128};
    std::vector<int> minimum_rollout_sizes {1000, 5000};
    std::vector<float> learning_rates {0.0000001, 0.00000001, 0.000000001};

    double tau = 0.95;                  //
    double ppo_clip = 0.2;              //
    int gradient_clip = 5;              //Clip gradient to try to prevent unstable learning
    float entropy_weight = 0.01;              //Clip gradient to try to prevent unstable learning

    int numProcessed = 0;
    for(auto discount_rate : discount_rates) {
        for(auto learning_round : learning_rounds) {
            for(auto mini_batch_number : mini_batch_numbers) {
                for(auto minimum_rollout_size : minimum_rollout_sizes) {
                    for(auto learning_rate : learning_rates) {
                        try {
                            std::cout << discount_rate << " " << learning_round << " " << mini_batch_number;
                            std::cout << " " << ppo_clip << " " << minimum_rollout_size << " " << learning_rate << std::endl;
                            if (numProcessed < 24) {
                                //Skipping combinations we'd done already
                                numProcessed++;
                                continue;
                            }

                            int numEpisodes = 1000;
                            std::cout << "NumProccesed: " << numProcessed << std::endl;
                            Agent agent(discount_rate, tau, learning_round, mini_batch_number, ppo_clip, minimum_rollout_size, learning_rate, entropy_weight);
                            ppo(agent, numEpisodes, numProcessed);
                        }
                        catch (const std::exception& e) {
                            std::cout << std::endl << "ERROR!" << std::endl;
                        }
                        numProcessed++;
                        std::cout << std::endl << "~~~~~~~~~~" << std::endl;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {

    //runGridSearch();

    int numEpisodes = 10000;
    int numProcessed = 0;

    float discount_rate = 0.99;
    float tau = 0.95;
    float learningRounds = 2;
    float mini_batch_number = 32;
    float ppo_clip = 0.2;
    float minimum_rollout_size = 3000;
    float learning_rate = 0.0000001;
    float entropy_weight = 0.01;

    Agent agent(discount_rate, tau, learningRounds, mini_batch_number, ppo_clip, minimum_rollout_size, learning_rate, entropy_weight);
    loadWeights(agent);       
    ppo(agent, numEpisodes, numProcessed);


    return 0;
}
