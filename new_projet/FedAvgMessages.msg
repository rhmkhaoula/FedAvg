// FedAvgMessages.msg
// Message definitions for FedAvg algorithm in OMNeT++

cplusplus {{
#include "inet/common/packet/chunk/Chunk.h"
#include "inet/common/INETDefs.h"
#include <vector>
}}

namespace inet;

class noncobject std::vector<double>;

// This message represents a model update (weights and biases)
class FedAvgModelUpdate extends cObject {
    std::vector<double> weights;    // Model weights
    int uavId;                      // ID of the UAV that sent the update
    int numSamples;                 // Number of samples used for training
    int roundNumber;                // Training round number
    simtime_t trainingTime;         // Time spent on local training
}

// This message is used by the base station to initiate a training round
class FedAvgInitiateTraining extends cObject {
    int roundNumber;                // Current training round number
    std::vector<double> weights;    // Global model weights to train from
}

// This message is used by the base station to broadcast the updated global model
class FedAvgGlobalModel extends cObject {
    int roundNumber;                // Training round that just completed
    std::vector<double> weights;    // Updated global model weights
    double globalLoss;              // Global loss after aggregation
    double globalAccuracy;          // Global accuracy after aggregation
}