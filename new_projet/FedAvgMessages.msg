// FedAvgMessages.msg
// Message definitions for FedAvg algorithm in OMNeT++

import inet.common.INETDefs;
import inet.common.packet.chunk.Chunk;

cplusplus {{
#include <vector>
}}

namespace inet;

class FedAvgModelUpdate extends cObject {
    @customize(true);
    @descriptor(readonly);
    @fieldNameSuffix("_var");
    abstract double weights[] @getter(getWeights) @sizeGetter(getWeightsArraySize) @setter(setWeights);
    int uavId;                      // ID of the UAV that sent the update
    int numSamples;                 // Number of samples used for training
    int roundNumber;                // Training round number
    simtime_t trainingTime;         // Time spent on local training
}

cplusplus(FedAvgModelUpdate) {{
  public:
    typedef std::vector<double> WeightsVector;
    
    // Getters for std::vector field
    const WeightsVector& getWeights() const { return weights_var; }
    double getWeights(size_t k) const { return weights_var[k]; }
    void setWeights(const WeightsVector& weights) { weights_var = weights; }
    void setWeights(size_t k, double weight) { weights_var[k] = weight; }
    size_t getWeightsArraySize() const { return weights_var.size(); }
    
  private:
    WeightsVector weights_var;
}}

class FedAvgInitiateTraining extends cObject {
    @customize(true);
    @descriptor(readonly);
    @fieldNameSuffix("_var");
    int roundNumber;                // Current training round number
    abstract double weights[] @getter(getWeights) @sizeGetter(getWeightsArraySize) @setter(setWeights);
}

cplusplus(FedAvgInitiateTraining) {{
  public:
    typedef std::vector<double> WeightsVector;
    
    // Getters for std::vector field
    const WeightsVector& getWeights() const { return weights_var; }
    double getWeights(size_t k) const { return weights_var[k]; }
    void setWeights(const WeightsVector& weights) { weights_var = weights; }
    void setWeights(size_t k, double weight) { weights_var[k] = weight; }
    size_t getWeightsArraySize() const { return weights_var.size(); }
    
  private:
    WeightsVector weights_var;
}}

class FedAvgGlobalModel extends cObject {
    @customize(true);
    @descriptor(readonly);
    @fieldNameSuffix("_var");
    int roundNumber;                // Training round that just completed
    abstract double weights[] @getter(getWeights) @sizeGetter(getWeightsArraySize) @setter(setWeights);
    double globalLoss;              // Global loss after aggregation
    double globalAccuracy;          // Global accuracy after aggregation
}

cplusplus(FedAvgGlobalModel) {{
  public:
    typedef std::vector<double> WeightsVector;
    
    // Getters for std::vector field
    const WeightsVector& getWeights() const { return weights_var; }
    double getWeights(size_t k) const { return weights_var[k]; }
    void setWeights(const WeightsVector& weights) { weights_var = weights; }
    void setWeights(size_t k, double weight) { weights_var[k] = weight; }
    size_t getWeightsArraySize() const { return weights_var.size(); }
    
  private:
    WeightsVector weights_var;
}}
