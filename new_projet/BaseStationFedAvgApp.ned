import inet.applications.contract.IApp;

simple BaseStationFedAvgApp like IApp
{
    parameters:
        string interfaceTableModule;
        double startTime @unit(s) = default(1s);
        double aggregationInterval @unit(s) = default(10s);
        double roundInterval @unit(s) = default(20s);
        int localPort;
        int clientPort;
        int minUpdatesForAggregation = default(3);
        int totalClients = default(5);
        double stopOperationExtraTime @unit(s) = default(2s);
        double stopOperationTimeout @unit(s) = default(2s);
        
        @display("i=block/app");
        @signal[rcvdPk](type=inet::Packet);
        @signal[aggregationCompleted](type=int);
        @signal[globalLoss](type=double);
        @signal[globalAccuracy](type=double);
        @statistic[rcvdPk](title="packets received"; source=rcvdPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[aggregationCompleted](title="aggregations completed"; source=aggregationCompleted; record=vector; interpolationmode=none);
        @statistic[globalLoss](title="global loss"; source=globalLoss; record=vector; interpolationmode=none);
        @statistic[globalAccuracy](title="global accuracy"; source=globalAccuracy; record=vector; interpolationmode=none);
        
    gates:
        input socketIn;
        output socketOut;
}