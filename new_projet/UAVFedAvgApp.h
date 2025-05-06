#ifndef __UAVFEDAVGAPP_H
#define __UAVFEDAVGAPP_H

#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/lifecycle/LifecycleOperation.h"
#include "inet/common/packet/Packet.h"
#include "FedAvgModel.h"
#include "FedAvgMessages_m.h"

using namespace omnetpp;
using namespace inet;

class UAVFedAvgApp : public ApplicationBase, public UdpSocket::ICallback {
  protected:
    // Configuration
    int localPort = -1;
    int destPort = -1;
    L3Address destAddress;

    // Socket and timers
    UdpSocket socket;
    cMessage *sensorDataTimer = nullptr;
    cMessage *trainingTimer = nullptr;
    simtime_t sensorInterval;
    simtime_t trainingInterval;

    // Federated Learning components
    FedAvgModel localModel;
    int currentRound = 0;
    int dataCollectionSize = 100; // Number of samples to collect before training
    bool trainingInProgress = false;

    // Simulated sensor data storage
    std::vector<std::vector<double>> localData;

    // Statistics
    int numSent = 0;
    int numReceived = 0;
    int numTrainingRounds = 0;
    static simsignal_t sentPkSignal;
    static simsignal_t rcvdPkSignal;
    static simsignal_t roundCompletedSignal;
    static simsignal_t trainingLossSignal;

  protected:
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;

    // Application methods
    virtual void sendSensorData();
    virtual void collectSensorData();
    virtual void performLocalTraining();
    virtual void sendModelUpdate();
    virtual void processGlobalModel(FedAvgGlobalModel* globalModel);
    virtual void startTrainingRound(FedAvgInitiateTraining* initMsg);

    // Socket methods
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    // LifecycleOperation
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:
    UAVFedAvgApp();
    virtual ~UAVFedAvgApp();
};

#endif
