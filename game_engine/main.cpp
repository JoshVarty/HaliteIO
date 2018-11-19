#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <deque>

#include "Halite.hpp"
#include "Constants.hpp"
#include "Generator.hpp"
#include "Halite.hpp"
#include "Logging.hpp"
#include "Replay.hpp"
#include "Snapshot.hpp"
#include "Agent.hpp"

#include "tclap/CmdLine.h"

void ppo(Agent myAgent, unsigned int numEpisodes) {
    std::vector<double> allScores;
    std::deque<double> lastHundredScores;

    for (uint i = 1; i < numEpisodes + 1; i++){

        double current_score = myAgent.step();
        allScores.push_back(current_score);
        //Keep track of the last 100 scores
        lastHundredScores.push_back(current_score);
        if(lastHundredScores.size() > 100) {
            lastHundredScores.pop_front();
        }

        double mean = 0;
        for (uint j = 0; j < lastHundredScores.size(); j++){
            mean += 0.01 * lastHundredScores[j];
        }

        if (i % 100 == 0) {
            //Every 100 episodes, display the mean reward
            std::cout << "Mean at step: " << i << ": " << mean;
        }

        //TODO: If network improves, save it.
    }
}

int main(int argc, char *argv[]) {
    auto &constants = hlt::Constants::get_mut();

    Agent agent;
    ppo(agent, 500);
}
