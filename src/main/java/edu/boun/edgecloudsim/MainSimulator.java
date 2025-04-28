package edu.boun.edgecloudsim;

import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.applications.RLTaskScheduler;


public class MainSimulator {
    public static void main(String[] args) {
        SimManager simManager = SimManager.getInstance();
        //simManager.initialize("config/edge_config.properties");
        
        RLTaskScheduler rlScheduler = new RLTaskScheduler("some_scenario", "RLTaskScheduler");
        //simManager.setOrchestrator(rlScheduler);
        
        //simManager.startSimulation();
    }
}
