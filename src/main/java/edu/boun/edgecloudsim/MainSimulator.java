package edu.boun.edgecloudsim;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;

import edu.boun.edgecloudsim.applications.RLTaskScheduler;
import edu.boun.edgecloudsim.core.ScenarioFactory;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_client.mobile_processing_unit.DefaultMobileServerManager;
import edu.boun.edgecloudsim.edge_client.DefaultMobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.FuzzyMobileDeviceManager;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager;
import edu.boun.edgecloudsim.mobility.NomadicMobility;
import edu.boun.edgecloudsim.mobility.MobilityModel;
import edu.boun.edgecloudsim.network.MM1Queue;
import edu.boun.edgecloudsim.task_generator.IdleActiveLoadGenerator;
import edu.boun.edgecloudsim.task_generator.LoadGeneratorModel;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.SimUtils;

public class MainSimulator {
    public static void main(String[] args) {
        // 1) disable CloudSim's internal log, enable ours
        Log.disable();                
        SimLogger.enablePrintLog();

        // 2) parse arguments or fall back to defaults
        String configFile        = args.length > 0 ? args[0] : "rl_config.properties";
        String edgeDevicesFile   = args.length > 1 ? args[1] : "edge_devices.xml";
        String applicationsFile  = args.length > 2 ? args[2] : "applications.xml";
        String outputFolder      = args.length > 3 ? args[3] : "sim_results";

        // 3) load the three XML/properties files into SimSettings
        SimSettings settings = SimSettings.getInstance();
        if (!settings.initialize(configFile, edgeDevicesFile, applicationsFile)) {
            SimLogger.printLine("ERROR: Failed to initialize simulation settings");
            System.exit(0);
        }
        // if file-logging enabled, clear old results
        if (settings.getFileLoggingEnabled()) {
            SimLogger.enableFileLog();
            SimUtils.cleanOutputFolder(outputFolder);
            String policyName = settings.getOrchestratorPolicies()[0];
            SimLogger.getInstance().simStarted(outputFolder, policyName);
        }

        // 4) record start time
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        Date start = Calendar.getInstance().getTime();
        SimLogger.printLine("Simulation started at " + df.format(start));

        // 5) choose one scenario / one policy (you can loop over settings.getSimulationScenarios() etc.)
        final int   numDevices   = settings.getMinNumOfMobileDev();
        final double simTime     = settings.getSimulationTime();
        final String scenario    = settings.getSimulationScenarios()[0];
        final String policyName  = "RLTaskScheduler";

        try {
            // 6) initialize CloudSim
            CloudSim.init(1, Calendar.getInstance(), false, 0.01);  // num_user=1, trace=false :contentReference[oaicite:0]{index=0}

            // 7) build a ScenarioFactory that hands back your RLTaskScheduler
            ScenarioFactory factory = new ScenarioFactory() {
                @Override
                public LoadGeneratorModel getLoadGeneratorModel() {
                    // use the default IdleActive generator
                    return new IdleActiveLoadGenerator(numDevices, simTime, scenario);
                }
                @Override
                public RLTaskScheduler getEdgeOrchestrator() {
                    // instantiate your scheduler; this creates rl_experiences.csv
                    return new RLTaskScheduler(scenario, policyName);
                }
                @Override
                public MobilityModel getMobilityModel() {
                    return new NomadicMobility(numDevices, simTime);
                }
                @Override
                public MM1Queue getNetworkModel() {
                    return new MM1Queue(numDevices, scenario);
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
                    return new DefaultMobileDeviceManager() {
                    };
                }
            };

            // 8) create and run the simulation
            SimManager manager = new SimManager(factory, numDevices, scenario, policyName);
            manager.startSimulation();

        } catch (Exception e) {
            SimLogger.printLine("Simulation failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }

        // 9) record end time
        Date end = Calendar.getInstance().getTime();
        SimLogger.printLine("Simulation finished at " + df.format(end));
    }
}
