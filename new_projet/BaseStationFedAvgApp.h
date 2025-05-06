#include "BaseStationFedAvgApp.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TimeTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/common/packet/chunk/cPacketChunk.h"

Define_Module(BaseStationFedAvgApp);

simsignal_t BaseStationFedAvgApp::rcvdPkSignal = registerSignal("rcvdPk");
simsignal_t BaseStationFedAvgApp::aggregationCompletedSignal = registerSignal("aggregationCompleted");
simsignal_t BaseStationFedAvgApp::globalLossSignal = registerSignal("globalLoss");
simsignal_t BaseStationFedAvgApp::globalAccuracySignal = registerSignal("globalAccuracy");

BaseStationFedAvgApp::BaseStationFedAvgApp() : globalModel(10, 2) {
}

BaseStationFedAvgApp::~BaseStationFedAvgApp() {
    cancelAndDelete(aggregationTimer);
    cancelAndDelete(roundStartTimer);

    // Clean up any stored model updates
    for (auto& entry : receivedUpdates) {
        delete entry.second;
    }
    receivedUpdates.clear();
}

void BaseStationFedAvgApp::initialize(int stage) {
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        clientPort = par("clientPort");
        aggregationInterval = par("aggregationInterval");
        roundInterval = par("roundInterval");
        minUpdatesForAggregation = par("minUpdatesForAggregation");
        totalClients = par("totalClients");

        numReceived = 0;
        numRoundsCompleted = 0;
        numModelUpdatesReceived = 0;
        WATCH(numReceived);
        WATCH(numRoundsCompleted);
        WATCH(numModelUpdatesReceived);
        WATCH(currentRound);
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        aggregationTimer = new cMessage("aggregationTimer");
        roundStartTimer = new cMessage("roundStartTimer");

        socket.setOutputGate(gate("socketOut"));
        socket.bind(localPort);
        socket.setCallback(this);

        // Schedule the first training round to start
        scheduleAt(simTime() + par("startTime"), roundStartTimer);
    }
}

void BaseStationFedAvgApp::handleMessageWhenUp(cMessage *msg) {
    if (msg->isSelfMessage()) {
        if (msg == aggregationTimer) {
            aggregateModels();
        }
        else if (msg == roundStartTimer) {
            startNewRound();
        }
    }
    else
        socket.processMessage(msg);
}

void BaseStationFedAvgApp::startNewRound() {
    EV_INFO << "Starting new training round " << currentRound << endl;

    // Reset for new round
    roundInProgress = true;
    for (auto& entry : receivedUpdates) {
        delete entry.second;
    }
    receivedUpdates.clear();

    // Tell clients to start training
    broadcastInitiateTraining();

    // Schedule aggregation after some time
    scheduleAt(simTime() + aggregationInterval, aggregationTimer);
}

void BaseStationFedAvgApp::broadcastInitiateTraining() {
    // Create initiate training message
    FedAvgInitiateTraining* initMsg = new FedAvgInitiateTraining();
    initMsg->setRoundNumber(currentRound);
    initMsg->setWeights(globalModel.getWeights());

    // Create packet
    char msgName[32];
    sprintf(msgName, "InitTraining-Round-%d", currentRound);
    Packet *packet = new Packet(msgName);

    // Add message as packet chunk
    auto packetChunk = new cPacketChunk(initMsg);
    packet->insertAtBack(std::shared_ptr<cPacketChunk>(packetChunk));

    // Broadcast to all registered clients
    if (clientAddresses.empty()) {
        // If no clients registered yet, broadcast to network
        socket.sendTo(packet, L3Address(), clientPort);
        EV_INFO << "Broadcasting training initiation (round " << currentRound << ") to all potential clients" << endl;
    } else {
        // Send to each registered client
        for (const auto& client : clientAddresses) {
            socket.sendTo(packet->dup(), client.first, clientPort);
        }
        delete packet; // Delete original after dups sent
        EV_INFO << "Sent training initiation to " << clientAddresses.size() << " registered clients" << endl;
    }
}

void BaseStationFedAvgApp::aggregateModels() {
    EV_INFO << "Aggregating models for round " << currentRound << endl;

    // Check if we have enough updates
    if (receivedUpdates.size() < minUpdatesForAggregation) {
        EV_WARN << "Not enough model updates received. Got " << receivedUpdates.size()
                << ", need " << minUpdatesForAggregation << ". Extending aggregation time." << endl;

        // Reschedule aggregation
        scheduleAt(simTime() + aggregationInterval/2, aggregationTimer);
        return;
    }

    // Implement FedAvg: weighted average of models based on number of samples
    // Get total number of samples across all clients
    int totalSamples = 0;
    for (const auto& entry : receivedUpdates) {
        totalSamples += entry.second->getNumSamples();
    }

    if (totalSamples == 0) {
        EV_ERROR << "Error: Total samples is 0, cannot perform weighted average" << endl;
        return;
    }

    // Get the size of weights vector (assuming all have same size)
    const auto& firstWeights = receivedUpdates.begin()->second->getWeights();
    int weightSize = firstWeights.size();

    // Initialize new weights vector
    std::vector<double> aggregatedWeights(weightSize, 0.0);

    // Perform weighted average
    for (const auto& entry : receivedUpdates) {
        const FedAvgModelUpdate* update = entry.second;
        const auto& weights = update->getWeights();
        double weight = static_cast<double>(update->getNumSamples()) / totalSamples;

        for (size_t i = 0; i < weightSize; i++) {
            aggregatedWeights[i] += weights[i] * weight;
        }
    }

    // Update global model
    globalModel.setWeights(aggregatedWeights);

    // Simulate evaluating the global model
    double globalAccuracy = globalModel.evaluate(totalSamples);
    double globalLoss = 1.0 - globalAccuracy; // Simple inverse for demonstration

    // Emit statistics
    emit(globalAccuracySignal, globalAccuracy);
    emit(globalLossSignal, globalLoss);
    emit(aggregationCompletedSignal, currentRound);

    EV_INFO << "Model aggregation complete. Round: " << currentRound
            << ", Global Accuracy: " << globalAccuracy
            << ", Global Loss: " << globalLoss << endl;

    // Broadcast the new global model
    broadcastGlobalModel();

    // Complete the round
    numRoundsCompleted++;
    roundInProgress = false;

    // Schedule next round
    currentRound++;
    scheduleAt(simTime() + roundInterval, roundStartTimer);
}

void BaseStationFedAvgApp::broadcastGlobalModel() {
    // Create global model message
    FedAvgGlobalModel* globalModelMsg = new FedAvgGlobalModel();
    globalModelMsg->setRoundNumber(currentRound);
    globalModelMsg->setWeights(globalModel.getWeights());

    // Simple simulation of metrics
    double accuracy = globalModel.evaluate(1000); // Simulate evaluation
    globalModelMsg->setGlobalAccuracy(accuracy);
    globalModelMsg->setGlobalLoss(1.0 - accuracy);

    // Create packet
    char msgName[32];
    sprintf(msgName, "GlobalModel-Round-%d", currentRound);
    Packet *packet = new Packet(msgName);

    // Add message as packet chunk
    auto packetChunk = new cPacketChunk(globalModelMsg);
    packet->insertAtBack(std::shared_ptr<cPacketChunk>(packetChunk));

    // Broadcast to all registered clients
    if (clientAddresses.empty()) {
        // If no clients registered yet, broadcast to network
        socket.sendTo(packet, L3Address(), clientPort);
        EV_INFO << "Broadcasting global model (round " << currentRound << ") to all potential clients" << endl;
    } else {
        // Send to each registered client
        for (const auto& client : clientAddresses) {
            socket.sendTo(packet->dup(), client.first, clientPort);
        }
        delete packet; // Delete original after dups sent
        EV_INFO << "Sent global model to " << clientAddresses.size() << " clients" << endl;
    }
}

void BaseStationFedAvgApp::socketDataArrived(UdpSocket *socket, Packet *packet) {
    // Process incoming packets from UAVs
    auto addressInd = packet->getTag<L3AddressInd>();
    L3Address srcAddr = addressInd->getSrcAddress();

    // Calculate end-to-end delay
    auto creationTimeTag = packet->getTag<CreationTimeTag>();
    simtime_t delay = simTime() - creationTimeTag->getCreationTime();

    EV_INFO << "Received packet " << packet->getName() << " from UAV at "
            << srcAddr.str() << ". Delay: " << delay << "s" << endl;

    // Update statistics
    numReceived++;
    emit(rcvdPkSignal, packet);

    // Check if it's a model update
    cPacketChunk *chunk = dynamic_cast<cPacketChunk *>(packet->peekAtFront().get());
    if (chunk) {
        cPacket *innerPacket = chunk->getPacket();

        if (FedAvgModelUpdate *modelUpdate = dynamic_cast<FedAvgModelUpdate *>(innerPacket)) {
            // Register client if not already registered
            if (clientAddresses.find(srcAddr) == clientAddresses.end()) {
                clientAddresses[srcAddr] = modelUpdate->getUavId();
                EV_INFO << "Registered new client: " << srcAddr.str() << " with ID " << modelUpdate->getUavId() << endl;
            }

            // Process the model update
            processModelUpdate(modelUpdate, srcAddr);

            // Take ownership of modelUpdate from the packet to store it
            auto modelUpdateCopy = modelUpdate->dup();
            delete packet;
            return;
        }
    }

    delete packet;
}

void BaseStationFedAvgApp::processModelUpdate(FedAvgModelUpdate* update, L3Address senderAddr) {
    int clientId = update->getUavId();

    EV_INFO << "Processing model update from UAV ID " << clientId
            << " for round " << update->getRoundNumber()
            << " with " << update->getNumSamples() << " samples" << endl;

    // Only process if it's for the current round
    if (update->getRoundNumber() == currentRound && roundInProgress) {
        // Store the update (replacing any previous update from this client)
        if (receivedUpdates.find(clientId) != receivedUpdates.end()) {
            delete receivedUpdates[clientId]; // Delete old update
        }

        receivedUpdates[clientId] = update->dup(); // Store a copy
        numModelUpdatesReceived++;

        EV_INFO << "Stored model update. Now have " << receivedUpdates.size()
                << " updates for round " << currentRound << endl;

        // If we have received updates from all clients, we can aggregate early
        if (receivedUpdates.size() >= totalClients) {
            EV_INFO << "Received updates from all clients. Aggregating early." << endl;
            cancelEvent(aggregationTimer);
            scheduleAt(simTime() + 0.1, aggregationTimer); // Aggregate soon
        }
    } else {
        EV_WARN << "Received model update for wrong round. Current round: "
                << currentRound << ", update round: " << update->getRoundNumber() << endl;
    }
}

void BaseStationFedAvgApp::socketErrorArrived(UdpSocket *socket, Indication *indication) {
    EV_WARN << "Socket error: " << indication->getName() << endl;
    delete indication;
}

void BaseStationFedAvgApp::socketClosed(UdpSocket *socket) {
    if (operationalState == State::STOPPING_OPERATION) {
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
    }
}

void BaseStationFedAvgApp::handleStartOperation(LifecycleOperation *operation) {
    socket.setOutputGate(gate("socketOut"));
    socket.bind(localPort);
    socket.setCallback(this);

    // Start federated learning process
    scheduleAt(simTime() + par("startTime"), roundStartTimer);
}

void BaseStationFedAvgApp::handleStopOperation(LifecycleOperation *operation) {
    cancelEvent(aggregationTimer);
    cancelEvent(roundStartTimer);
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void BaseStationFedAvgApp::handleCrashOperation(LifecycleOperation *operation) {
    cancelEvent(aggregationTimer);
    cancelEvent(roundStartTimer);
    socket.destroy();
}

void BaseStationFedAvgApp::finish() {
    ApplicationBase::finish();

    EV_INFO << "Base Station FedAvg Application finished." << endl;
    EV_INFO << "Received: " << numReceived << " packets in total." << endl;
    EV_INFO << "Completed " << numRoundsCompleted << " federated learning rounds." << endl;
    EV_INFO << "Received " << numModelUpdatesReceived << " model updates from clients." << endl;

    EV_INFO << "Registered clients:" << endl;
    for (auto& pair : clientAddresses) {
        EV_INFO << "  UAV at " << pair.first.str() << " with ID " << pair.second << endl;
    }
}
