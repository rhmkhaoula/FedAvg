import inet.applications.contract.IApp;

simple UAVFedAvgApp like IApp
{
    parameters:
        string interfaceTableModule;
        double startTime @unit(s) = default(1s);
        double sensorInterval @unit(s) = default(1s);
        double trainingInterval @unit(s) = default(5s);
        int localPort = default(-1);
        int destPort;
        int messageLength @unit(B) = default(100B);
        int dataCollectionSize = default(100);
        string destAddresses = default("");
        double stopOperationExtraTime @unit(s) = default(2s);
        double stopOperationTimeout @unit(s) = default(2s);
        
        @display("i=block/app");
        @signal[sentPk](type=inet::Packet);
        @signal[rcvdPk](type=inet::Packet);
        @signal[roundCompleted](type=int);
        @signal[trainingLoss](type=double);
        @statistic[sentPk](title="packets sent"; source=sentPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[rcvdPk](title="packets received"; source=rcvdPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[roundCompleted](title="completed rounds"; source=roundCompleted; record=vector; interpolationmode=none);
        @statistic[trainingLoss](title="training loss"; source=trainingLoss; record=vector; interpolationmode=none);
        
    gates:
        input socketIn;
        output socketOut;
}