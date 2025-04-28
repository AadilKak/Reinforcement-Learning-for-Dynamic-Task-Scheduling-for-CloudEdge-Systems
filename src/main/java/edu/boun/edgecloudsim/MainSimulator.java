package edu.boun.edgecloudsim;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.applications.RLTaskScheduler;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.task_generator.IdleActiveLoadGenerator;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.network.MM1Queue;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager;

public class MainSimulator {
    public static void main(String[] args) throws Exception {
        // 1) Load simulation settings
        SimSettings.
        createInstance("rl_config.properties");

        // 2) Number of mobile devices
        int numDevices = SimSettings.getInstance().getNumOfMobileDevice();

        // 3) Scenario factory with custom components
        ScenarioFactory factory = new ScenarioFactory() {
            @Override
            public LoadGeneratorModel getLoadGeneratorModel() {
                return new IdleActiveLoadGenerator();
            }

            @Override
            public EdgeOrchestrator getEdgeOrchestrator() {
                return new RLTaskScheduler("MyScenario", "RLTaskScheduler");
            }

            @Override
            public MobilityModel getMobilityModel() {
                return new NomadicMobility();
            }

            @Override
            public NetworkModel getNetworkModel() {
                return new MM1Queue();
            }

            @Override
            public DefaultEdgeServerManager getEdgeServerManager() {
                return new DefaultEdgeServerManager();
            }

            @Override
            public DefaultCloudServerManager getCloudServerManager() {
                return new DefaultCloudServerManager();
            }

            @Override
            public DefaultMobileServerManager getMobileServerManager() {
                return new DefaultMobileServerManager();
            }

            @Override
            public DefaultMobileDeviceManager getMobileDeviceManager() throws Exception {
                return new DefaultMobileDeviceManager();
            }
        };

        // 4) Initialize and start the simulation
        SimManager simManager = new SimManager(
            factory,
            numDevices,
            "MyScenario",        // Simulation scenario name
            "RLTaskScheduler"    // Orchestrator policy name
        );
        simManager.startSimulation();
    }
}
