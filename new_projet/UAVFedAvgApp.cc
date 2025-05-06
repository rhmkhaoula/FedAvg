#include "UAVFedAvgApp.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TimeTag_m.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/common/packet/chunk/cPacketChunk.h"

Define_Module(UAVFedAvgApp);

simsignal_t UAVFedAvgApp::sentPkSignal = registerSignal("sentPk");
simsignal_t UAVFedAvgApp::rcvdPkSignal = registerSignal("rcvdPk");
simsignal_t UAVFedAvgApp::roundCompletedSignal = registerSignal("roundCompleted");
simsignal_t UAVFedAvgApp::trainingLossSignal = registerSignal("trainingLoss");

UAVFedAvgApp::UAVFedAvgApp() : localModel(10, 2) {
}

UAVFedAvgApp::~UAVFedAvgApp() {
    cancelAndDelete(sensorDataTimer);
    cancelAndDelete(trainingTimer);
}

void UAVFedAvgApp::initialize(int stage) {
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        sensorInterval = par("sensorInterval");
        trainingInterval = par("trainingInterval");
        localPort = par("localPort");
        destPort = par("destPort");
        dataCollectionSize = par("dataCollectionSize");

        // Initialize statistics
        numSent = 0;
        numReceived = 0;
        numTrainingRounds = 0;
        WATCH(numSent);
        WATCH(numReceived);
        WATCH(numTrainingRounds);
        WATCH(currentRound);
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        sensorDataTimer = new cMessage("sensorDataTimer");
        trainingTimer = new cMessage("trainingTimer");

        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        socket.setCallback(this);

        const char *destAddrs = par("destAddresses");
        cStringTokenizer tokenizer(destAddrs);
        const char *token;

        while ((token = tokenizer.nextToken()) != nullptr) {
            L3AddressResolver().tryResolve(token, destAddress);
            if (destAddress.isUnspecified())
                EV_ERROR << "Cannot resolve destination address: " << token << endl;
            break;
        }

        if (destAddress.isUnspecified()) {
            EV_WARN << "No destination address specified, app won't send packets" << endl;
        }
        else if (operationalState == State::OPERATING) {
            // Start sensor data collection
            scheduleAt(simTime() + par("startTime"), sensorDataTimer);
        }
    }
}

void UAVFedAvgApp::handleMessageWhenUp(cMessage *msg) {
    if (msg->isSelfMessage()) {
        if (msg == sensorDataTimer) {
            sendSensorData();
            scheduleAt(simTime() + sensorInterval, sensorDataTimer);
        }
        else if (msg == trainingTimer) {
            performLocalTraining();
        }
    }
    else
        socket.processMessage(msg);
}

void UAVFedAvgApp::collectSensorData() {
    // Simulate sensor data collection
    std::vector<double> dataPoint(10); // 10 features per sample

    // Generate random sensor data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    for (auto& value : dataPoint) {
        value = dist(gen);
    }

    // Store data locally
    localData.push_back(dataPoint);

    EV_INFO << "Collected sensor data: sample #" << localData.size() << endl;

    // If we've collected enough data, we can train
    if (localData.size() >= dataCollectionSize && !trainingInProgress) {
        // Schedule training if not already in progress
        EV_INFO << "Enough data collected, scheduling local training" << endl;
        scheduleAt(simTime() + 0.01, trainingTimer);
    }
}

void UAVFedAvgApp::performLocalTraining() {
    EV_INFO << "Starting local training on " << localData.size() << " samples" << endl;
    trainingInProgress = true;

    // Train the local model
    auto [loss, samples] = localModel.train(localData.size());

    EV_INFO << "Local training completed. Loss: " << loss << endl;
    emit(trainingLossSignal, loss);

    // Send model update to base station
    sendModelUpdate();

    // Clear training flag
    trainingInProgress = false;
}

void UAVFedAvgApp::sendModelUpdate() {
    // Create model update message
    FedAvgModelUpdate* modelUpdate = new FedAvgModelUpdate();
    modelUpdate->setUavId(getId());
    modelUpdate->setWeights(localModel.getWeights());
    modelUpdate->setNumSamples(localData.size());
    modelUpdate->setRoundNumber(currentRound);
    modelUpdate->setTrainingTime(trainingInterval);

    // Create packet to send
    char msgName[32];
    sprintf(msgName, "ModelUpdate-%d-Round-%d", getId(), currentRound);
    Packet *packet = new Packet(msgName);

    // Add creation time tag
    auto creationTimeTag = packet->addTag<CreationTimeTag>();
    creationTimeTag->setCreationTime(simTime());

    // Add model update to packet
    auto packetChunk = new cPacketChunk(modelUpdate);
    packet->insertAtBack(std::shared_ptr<cPacketChunk>(packetChunk));

    // Send model update to base station
    socket.sendTo(packet, destAddress, destPort);

    EV_INFO << "Sent model update to base station. Round: " << currentRound << endl;

    numSent++;
    emit(sentPkSignal, packet);
}

void UAVFedAvgApp::sendSensorData() {
    collectSensorData();

    // We still send regular sensor data for monitoring purposes
    char msgName[32];
    sprintf(msgName, "UAVSensorData-%d", numSent);

    // Create packet with sensor data
    Packet *packet = new Packet(msgName);
    auto creationTimeTag = packet->addTag<CreationTimeTag>();
    creationTimeTag->setCreationTime(simTime());

    // Add sensor data payload
    const auto& payload = makeShared<ByteCountChunk>(B(par("messageLength")));
    packet->insertAtBack(payload);

    // Send to base station
    socket.sendTo(packet, destAddress, destPort);

    numSent++;
    emit(sentPkSignal, packet);
}

void UAVFedAvgApp::socketDataArrived(UdpSocket *socket, Packet *packet) {
    // Process incoming packets from base station
    EV_INFO << "Received packet from base station: " << packet->getName() << endl;

    numReceived++;
    emit(rcvdPkSignal, packet);

    // Check if it's a FedAvg message
    cPacketChunk *chunk = dynamic_cast<cPacketChunk *>(packet->peekAtFront().get());
    if (chunk) {
        cPacket *innerPacket = chunk->getPacket();

        // Check message type
        if (FedAvgInitiateTraining *initMsg = dynamic_cast<FedAvgInitiateTraining *>(innerPacket)) {
            // Base station wants us to start a new training round
            startTrainingRound(initMsg);
        }
        else if (FedAvgGlobalModel *globalModel = dynamic_cast<FedAvgGlobalModel *>(innerPacket)) {
            // Base station sent updated global model
            processGlobalModel(globalModel);
        }
    }

    delete packet;
}

void UAVFedAvgApp::startTrainingRound(FedAvgInitiateTraining* initMsg) {
    // Update current round
    currentRound = initMsg->getRoundNumber();

    // Update local model with global weights
    localModel.setWeights(initMsg->getWeights());

    EV_INFO << "Starting training round " << currentRound << endl;

    // Schedule local training
    if (!trainingInProgress) {
        scheduleAt(simTime() + 0.01, trainingTimer);
    }
}

void UAVFedAvgApp::processGlobalModel(FedAvgGlobalModel* globalModel) {
    // Update local model with new global weights
    localModel.setWeights(globalModel->getWeights());

    EV_INFO << "Updated local model with global weights. Round: " <<
        globalModel->getRoundNumber() <<
        ", Global Loss: " << globalModel->getGlobalLoss() <<
        ", Global Accuracy: " << globalModel->getGlobalAccuracy() << endl;

    // Emit signal for completed round
    emit(roundCompletedSignal, globalModel->getRoundNumber());

    // Update round number
    currentRound = globalModel->getRoundNumber() + 1;

    // If we have enough data, schedule next training round
    if (localData.size() >= dataCollectionSize && !trainingInProgress) {
        scheduleAt(simTime() + trainingInterval, trainingTimer);
    }
}

void UAVFedAvgApp::socketErrorArrived(UdpSocket *socket, Indication *indication) {
    EV_WARN << "Socket error: " << indication->getName() << endl;
    delete indication;
}

void UAVFedAvgApp::socketClosed(UdpSocket *socket) {
    if (operationalState == State::STOPPING_OPERATION) {
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
    }
}

void UAVFedAvgApp::handleStartOperation(LifecycleOperation *operation) {
    socket.setOutputGate(gate("socketOut"));
    socket.bind(localPort);
    socket.setCallback(this);

    if (!destAddress.isUnspecified()) {
        sensorDataTimer = new cMessage("sensorDataTimer");
        scheduleAt(simTime() + par("startTime"), sensorDataTimer);
    }
}

void UAVFedAvgApp::handleStopOperation(LifecycleOperation *operation) {
    cancelEvent(sensorDataTimer);
    cancelEvent(trainingTimer);
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void UAVFedAvgApp::handleCrashOperation(LifecycleOperation *operation) {
    cancelEvent(sensorDataTimer);
    cancelEvent(trainingTimer);
    socket.destroy();
}

void UAVFedAvgApp::finish() {
    ApplicationBase::finish();
    EV_INFO << "UAV FedAvg Application finished. Sent: " << numSent << " packets, Received: " << numReceived << " packets." << endl;
    EV_INFO << "Completed " << numTrainingRounds << " training rounds." << endl;
}
