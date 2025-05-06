#ifndef __FEDAVGMODEL_H
#define __FEDAVGMODEL_H

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

// A simple neural network model for demonstration purposes
class FedAvgModel {
private:
    std::vector<double> weights;
    int inputSize;
    int outputSize;

    // Random number generator for simulated training
    std::mt19937 rng;

public:
    // Initialize model with random weights
    FedAvgModel(int inputSize = 10, int outputSize = 2)
        : inputSize(inputSize), outputSize(outputSize) {

        // Seed random number generator
        std::random_device rd;
        rng = std::mt19937(rd());

        // Initialize weights with small random values
        int totalWeights = inputSize * outputSize;
        weights.resize(totalWeights);

        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (int i = 0; i < totalWeights; i++) {
            weights[i] = dist(rng);
        }
    }

    // Set model weights directly
    void setWeights(const std::vector<double>& newWeights) {
        if (newWeights.size() != weights.size()) {
            throw std::runtime_error("Weight dimensions do not match");
        }
        weights = newWeights;
    }

    // Get model weights
    const std::vector<double>& getWeights() const {
        return weights;
    }

    // Simulate local training on data
    // Returns: pair(loss, number of samples used)
    std::pair<double, int> train(int numSamples) {
        // Simulate training by adding small perturbations to weights
        std::normal_distribution<double> dist(0.0, 0.01);

        for (auto& w : weights) {
            w += dist(rng);
        }

        // Simulate a decreasing loss value
        // (in real implementation, this would be calculated from actual training)
        double simulatedLoss = 1.0 / (1.0 + 0.1 * numSamples);

        return {simulatedLoss, numSamples};
    }

    // Predict function (simplified for simulation)
    std::vector<double> predict(const std::vector<double>& input) {
        if (input.size() != inputSize) {
            throw std::runtime_error("Input size mismatch");
        }

        std::vector<double> output(outputSize, 0.0);
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                output[i] += input[j] * weights[j * outputSize + i];
            }
        }

        return output;
    }

    // Evaluate model performance (simplified)
    double evaluate(int numSamples) {
        // Simulate evaluation with a random accuracy between 0.5 and 1.0
        // Higher values for more training samples
        std::uniform_real_distribution<double> dist(0.5, 1.0);
        double baseAccuracy = dist(rng);

        // Accuracy improves with more samples but plateaus
        return baseAccuracy * (1.0 - exp(-0.001 * numSamples));
    }
};

#endif
