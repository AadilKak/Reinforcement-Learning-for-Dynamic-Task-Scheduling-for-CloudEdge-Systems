package edu.boun.edgecloudsim.applications.rl_based_scheduler;

import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.utils.SimLogger;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Vm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 
 * RL-based Task Scheduler for EdgeCloudSim
 * This class implements a reinforcement learning based task scheduler
 * that can use either DQN or PPO algorithms
 * 
 * @author RL Task Scheduler
 * @version 1.0
 */
public class RLTaskScheduler extends EdgeOrchestrator {
    
    private static final int STATE_DIM = 25; // State dimension - adjust based on your environment
    private static final int ACTION_DIM = 10; // Action dimension - adjust based on your environment
    
    private String rlAlgorithm; // "DQN" or "PPO"
    private boolean usePretrainedModel;
    private String modelPath;
    private boolean isTrainingMode;
    private String pythonInterpreter;
    private double explorationRate; // Epsilon for exploration during training
    
    // Experience collection for training
    private List<Map<String, Object>> experiences;
    private String experienceFilePath;
    
    // Metrics tracking
    private int totalTasks;
    private int successfulTasks;
    private double totalDelay;
    private double totalEnergy;
    private int deadlinesMet;
    
    // Time and resource constraints
    private double networkBandwidth; // in Mbps
    private double maxTaskDeadline; // in ms
    
    /**
     * Constructor
     * 
     * @param _simScenario  Simulation scenario
     * @param _orchestratorPolicy Orchestrator policy
     */
    public RLTaskScheduler(String _simScenario, String _orchestratorPolicy) {
        super(_simScenario, _orchestratorPolicy);
        
        // Initialize from properties file
        loadConfig();
        
        // Initialize metrics
        totalTasks = 0;
        successfulTasks = 0;
        totalDelay = 0;
        totalEnergy = 0;
        deadlinesMet = 0;
        
        // Initialize experience collection
        if (isTrainingMode) {
            experiences = new ArrayList<>();
            experienceFilePath = "rl_experiences.csv";
            // Create a fresh experience file
            try {
                createExperienceFile();
            } catch (IOException e) {
                SimLogger.getInstance().simLog("Error creating experience file: " + e.getMessage());
            }
        }
        
        SimLogger.getInstance().simLog("RL-based Task Scheduler initialized");
        SimLogger.getInstance().simLog("Algorithm: " + rlAlgorithm);
        SimLogger.getInstance().simLog("Training mode: " + isTrainingMode);
        if (usePretrainedModel) {
            SimLogger.getInstance().simLog("Using pretrained model: " + modelPath);
        }
    }
    
    /**
     * Load configuration from properties file
     */
    private void loadConfig() {
        Properties prop = new Properties();
        try (FileInputStream input = new FileInputStream("rl_config.properties")) {
            prop.load(input);
            
            // RL algorithm configuration
            rlAlgorithm = prop.getProperty("rl.algorithm", "DQN");
            usePretrainedModel = Boolean.parseBoolean(prop.getProperty("rl.use_pretrained", "false"));
            modelPath = prop.getProperty("rl.model_path", "models/dqn_cloud_edge.pt");
            isTrainingMode = Boolean.parseBoolean(prop.getProperty("rl.training_mode", "true"));
            pythonInterpreter = prop.getProperty("rl.python_interpreter", "python3");
            explorationRate = Double.parseDouble(prop.getProperty("rl.exploration_rate", "0.1"));
            
            // Environment configuration
            networkBandwidth = Double.parseDouble(prop.getProperty("env.network_bandwidth", "100.0")); // Mbps
            maxTaskDeadline = Double.parseDouble(prop.getProperty("env.max_task_deadline", "500.0")); // ms
            
        } catch (IOException e) {
            SimLogger.getInstance().simLog("Config file not found or error loading. Using default values.");
            
            // Set default values
            rlAlgorithm = "DQN";
            usePretrainedModel = false;
            modelPath = "models/dqn_cloud_edge.pt";
            isTrainingMode = true;
            pythonInterpreter = "python3";
            explorationRate = 0.1;
            networkBandwidth = 100.0; // Mbps
            maxTaskDeadline = 500.0; // ms
        }
    }
    
    /**
     * Create a fresh experience file for training
     */
    private void createExperienceFile() throws IOException {
        FileWriter writer = new FileWriter(experienceFilePath);
        writer.write("state,action,reward,next_state,done\n");
        writer.close();
    }
    
    /**
     * Save an experience to file for offline training
     */
    private void saveExperience(double[] state, int action, double reward, double[] nextState, boolean done) {
        if (!isTrainingMode) return;
        
        try {
            FileWriter writer = new FileWriter(experienceFilePath, true);
            
            // Convert state array to string
            StringBuilder stateStr = new StringBuilder();
            for (double val : state) {
                stateStr.append(val).append(",");
            }
            
            // Convert next state array to string
            StringBuilder nextStateStr = new StringBuilder();
            for (double val : nextState) {
                nextStateStr.append(val).append(",");
            }
            
            // Format: state,action,reward,next_state,done
            writer.write(stateStr.toString() + action + "," + reward + "," + nextStateStr.toString() + (done ? "1" : "0") + "\n");
            writer.close();
            
        } catch (IOException e) {
            SimLogger.getInstance().simLog("Error saving experience: " + e.getMessage());
        }
    }
    
    /**
     * Get the current state representation
     */
    private double[] getState(Task task) {
        double[] state = new double[STATE_DIM];
        int stateIndex = 0;
        
        // Get environment information
        List<EdgeVM> edgeVMs = SimManager.getInstance().getEdgeServerManager().getVmList();
        List<CloudVM> cloudVMs = SimManager.getInstance().getCloudServerManager().getVmList();
        
        // 1. Edge server state (CPU utilization, queue length)