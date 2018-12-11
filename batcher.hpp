#ifndef BATCHER_H
#define BATCHER_H

#include <cstdlib>
#include <vector>
#include "types.hpp"

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

#endif
