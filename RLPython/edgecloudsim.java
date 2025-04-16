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
    
    // Manual configuration flags
    private boolean manualConfig;
    private double cpuThreshold;
    private double bandwidthThreshold;
    private double energyWeight;
    private double delayWeight;
    private double deadlineWeight;
    
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
        if (manualConfig) {
            SimLogger.getInstance().simLog("Manual configuration enabled");
            SimLogger.getInstance().simLog("CPU Threshold: " + cpuThreshold);
            SimLogger.getInstance().simLog("Bandwidth Threshold: " + bandwidthThreshold);
            SimLogger.getInstance().simLog("Energy Weight: " + energyWeight);
            SimLogger.getInstance().simLog("Delay Weight: " + delayWeight);
            SimLogger.getInstance().simLog("Deadline Weight: " + deadlineWeight);
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
            
            // Manual configuration
            manualConfig = Boolean.parseBoolean(prop.getProperty("manual.enabled", "false"));
            cpuThreshold = Double.parseDouble(prop.getProperty("manual.cpu_threshold", "70.0"));
            bandwidthThreshold = Double.parseDouble(prop.getProperty("manual.bandwidth_threshold", "50.0"));
            energyWeight = Double.parseDouble(prop.getProperty("manual.energy_weight", "0.3"));
            delayWeight = Double.parseDouble(prop.getProperty("manual.delay_weight", "0.4"));
            deadlineWeight = Double.parseDouble(prop.getProperty("manual.deadline_weight", "0.3"));
            
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
            
            // Default manual configuration
            manualConfig = false;
            cpuThreshold = 70.0;
            bandwidthThreshold = 50.0;
            energyWeight = 0.3;
            delayWeight = 0.4;
            deadlineWeight = 0.3;
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
        for (EdgeVM vm : edgeVMs) {
            if (stateIndex < STATE_DIM - 2) {
                state[stateIndex++] = vm.getCloudletScheduler().getCloudletExecList().size() / 10.0; // Queue size
                state[stateIndex++] = vm.getCpuUtilization() / 100.0; // CPU utilization (normalized)
            }
        }
        
        // 2. Cloud server state (CPU utilization, queue length)
        for (CloudVM vm : cloudVMs) {
            if (stateIndex < STATE_DIM - 2) {
                state[stateIndex++] = vm.getCloudletScheduler().getCloudletExecList().size() / 15.0; // Queue size
                state[stateIndex++] = vm.getCpuUtilization() / 100.0; // CPU utilization (normalized)
            }
        }
        
        // 3. Network state
        state[stateIndex++] = networkBandwidth / 200.0; // Normalized bandwidth
        
        // 4. Task properties
        state[stateIndex++] = task.getCloudletLength() / 50000.0; // Normalized task size
        state[stateIndex++] = task.getCloudletFileSize() / 10000.0; // Normalized input size
        state[stateIndex++] = task.getCloudletOutputSize() / 10000.0; // Normalized output size
        
        // 5. Task deadline (if available)
        if (task.getDeadline() > 0) {
            state[stateIndex++] = task.getDeadline() / maxTaskDeadline; // Normalized deadline
        } else {
            state[stateIndex++] = 1.0; // Default if no deadline
        }
        
        // Fill remaining state elements with zeros
        while (stateIndex < STATE_DIM) {
            state[stateIndex++] = 0.0;
        }
        
        return state;
    }
    
    /**
     * Select action using RL model or manual configuration
     */
    private int selectAction(double[] state) {
        // If manual configuration is enabled, use rule-based decision
        if (manualConfig) {
            return selectActionManually(state);
        }
        
        // If in training mode with exploration, use epsilon-greedy
        if (isTrainingMode && Math.random() < explorationRate) {
            return (int) (Math.random() * ACTION_DIM);
        }
        
        // Use RL model for prediction
        if (usePretrainedModel) {
            try {
                int action = callPythonModel(state);
                return action;
            } catch (Exception e) {
                SimLogger.getInstance().simLog("Error calling Python model: " + e.getMessage());
                // Fallback to manual selection
                return selectActionManually(state);
            }
        } else {
            // If no model and not in manual mode, use a simple heuristic
            return selectActionManually(state);
        }
    }
    
    /**
     * Select action using manual configuration rules
     */
    private int selectActionManually(double[] state) {
        List<EdgeVM> edgeVMs = SimManager.getInstance().getEdgeServerManager().getVmList();
        List<CloudVM> cloudVMs = SimManager.getInstance().getCloudServerManager().getVmList();
        
        // Extract relevant state information
        int numEdgeVMs = edgeVMs.size();
        int numCloudVMs = cloudVMs.size();
        
        // Calculate decision metrics
        double[] cpuLoads = new double[numEdgeVMs + numCloudVMs];
        double[] queueLengths = new double[numEdgeVMs + numCloudVMs];
        double networkStatus = state[numEdgeVMs * 2 + numCloudVMs * 2]; // Assuming network status position
        
        // Populate CPU loads and queue lengths from state
        for (int i = 0; i < numEdgeVMs; i++) {
            queueLengths[i] = state[i * 2];
            cpuLoads[i] = state[i * 2 + 1];
        }
        
        for (int i = 0; i < numCloudVMs; i++) {
            queueLengths[numEdgeVMs + i] = state[numEdgeVMs * 2 + i * 2];
            cpuLoads[numEdgeVMs + i] = state[numEdgeVMs * 2 + i * 2 + 1];
        }
        
        // Apply manual configuration rules
        double[] scores = new double[numEdgeVMs + numCloudVMs];
        
        for (int i = 0; i < scores.length; i++) {
            // Consider CPU load (lower is better)
            double cpuScore = (1.0 - cpuLoads[i]) * 100;
            
            // Consider queue length (lower is better)
            double queueScore = (1.0 - queueLengths[i]) * 100;
            
            // For cloud VMs, consider network status
            double networkScore = 100.0;
            if (i >= numEdgeVMs) {
                networkScore = networkStatus * 100;
            }
            
            // Calculate weighted score
            scores[i] = (cpuScore * energyWeight) + 
                         (queueScore * delayWeight) + 
                         (networkScore * deadlineWeight);
            
            // Apply threshold rules
            if (cpuLoads[i] > cpuThreshold / 100.0) {
                scores[i] *= 0.5; // Heavily penalize overloaded machines
            }
            
            if (i >= numEdgeVMs && networkStatus < bandwidthThreshold / 100.0) {
                scores[i] *= 0.7; // Penalize cloud when network is congested
            }
        }
        
        // Find the VM with the highest score
        int bestVM = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestVM = i;
            }
        }
        
        return bestVM;
    }
    
    /**
     * Calls Python model to predict the action for a state
     */
    private int callPythonModel(double[] state) throws IOException {
        // Create temporary file containing state
        String statePath = "temp_state.csv";
        FileWriter writer = new FileWriter(statePath);
        
        // Write state to CSV format
        for (double val : state) {
            writer.write(String.valueOf(val) + ",");
        }
        writer.close();
        
        // Prepare Python command
        String pythonCommand = pythonInterpreter + " predict_action.py";
        pythonCommand += " --model " + modelPath;
        pythonCommand += " --algorithm " + rlAlgorithm;
        pythonCommand += " --state " + statePath;
        
        // Execute Python process
        Process process = Runtime.getRuntime().exec(pythonCommand);
        
        try {
            // Wait for process to complete with timeout
            boolean completed = process.waitFor(5, TimeUnit.SECONDS);
            
            if (!completed) {
                process.destroyForcibly();
                throw new IOException("Python process timed out");
            }
            
            // Read result
            String result = new String(Files.readAllBytes(Paths.get("temp_action.txt"))).trim();
            return Integer.parseInt(result);
            
        } catch (InterruptedException e) {
            throw new IOException("Python process was interrupted: " + e.getMessage());
        } finally {
            // Clean up temporary files
            try {
                Files.deleteIfExists(Paths.get(statePath));
                Files.deleteIfExists(Paths.get("temp_action.txt"));
            } catch (IOException e) {
                SimLogger.getInstance().simLog("Error cleaning up temp files: " + e.getMessage());
            }
        }
    }
    
    /**
     * Calculate reward for an action
     */
    private double calculateReward(Task task, int vmIndex, double processingTime, boolean deadlineMet) {
        // Get environment information
        List<EdgeVM> edgeVMs = SimManager.getInstance().getEdgeServerManager().getVmList();
        
        // Calculate energy consumption
        double energy;
        if (vmIndex < edgeVMs.size()) {
            // Edge VM (lower energy for small tasks, higher for large tasks)
            energy = task.getCloudletLength() * 0.7 * (1 + (task.getCloudletOutputSize() / 10000.0));
        } else {
            // Cloud VM (includes network transfer energy)
            energy = task.getCloudletFileSize() * 0.8 + task.getCloudletLength() * 0.5;
        }
        
        // Time penalty
        double timeReward = -processingTime / 100.0;
        
        // Energy penalty
        double energyReward = -energy * 0.1;
        
        // Deadline reward
        double deadlineReward = deadlineMet ? 5.0 : -2.0;
        
        // Total reward
        return timeReward + energyReward + deadlineReward;
    }
    
    /**
     * Schedule task to VM implementation
     */
    @Override
    public int getDeviceToOffload(Task task) {
        totalTasks++;
        
        // Get current state
        double[] state = getState(task);
        
        // Select action
        int action = selectAction(state);
        
        // Map action to VM index
        int selectedVM = mapActionToVM(action);
        
        // Estimate processing time and deadline metrics
        double processingTime = estimateProcessingTime(task, selectedVM);
        boolean deadlineMet = task.getDeadline() <= 0 || processingTime <= task.getDeadline();
        
        // Update metrics
        if (deadlineMet) {
            deadlinesMet++;
        }
        totalDelay += processingTime;
        
        // If in training mode, record experience
        if (isTrainingMode) {
            // Calculate reward
            double reward = calculateReward(task, selectedVM, processingTime, deadlineMet);
            
            // Will get next state after task execution
            double[] nextState = getState(task);
            
            // Save experience for training
            saveExperience(state, action, reward, nextState, false);
        }
        
        // Log decision
        if (SimManager.getInstance().getSimulationManager().getSimulationTime() % 10000 == 0) {
            SimLogger.getInstance().simLog(String.format(
                "Task %d scheduled to VM %d. Success rate: %.2f%%",
                task.getCloudletId(), selectedVM, 
                (deadlinesMet * 100.0 / totalTasks)
            ));
        }
        
        return selectedVM;
    }
    
    /**
     * Map action index to VM index
     */
    private int mapActionToVM(int action) {
        List<EdgeVM> edgeVMs = SimManager.getInstance().getEdgeServerManager().getVmList();
        List<CloudVM> cloudVMs = SimManager.getInstance().getCloudServerManager().getVmList();
        
        int totalVMs = edgeVMs.size() + cloudVMs.size();
        
        // Ensure action is within valid range
        if (action >= totalVMs) {
            action = action % totalVMs;
        }
        
        return action;
    }
    
    /**
     * Estimate processing time for a task on a VM
     */
    private double estimateProcessingTime(Task task, int vmIndex) {
        List<EdgeVM> edgeVMs = SimManager.getInstance().getEdgeServerManager().getVmList();
        List<CloudVM> cloudVMs = SimManager.getInstance().getCloudServerManager().getVmList();
        
        // Process on edge
        if (vmIndex < edgeVMs.size()) {
            EdgeVM vm = edgeVMs.get(vmIndex);
            
            // Calculate processing time based on edge VM
            double loadFactor = 1 + (vm.getCpuUtilization() / 100.0);
            double queueDelay = vm.getCloudletScheduler().getCloudletExecList().size() * 0.5;
            return (task.getCloudletLength() / vm.getMips() * loadFactor) + queueDelay;
        } 
        // Process on cloud
        else {
            CloudVM vm = cloudVMs.get(vmIndex - edgeVMs.size());
            
            // Calculate network transfer time
            double transferTime = task.getCloudletFileSize() * (100 / networkBandwidth);
            
            // Calculate cloud processing time
            double loadFactor = 1 + (vm.getCpuUtilization() / 150.0);  // Cloud is faster
            double queueDelay = vm.getCloudletScheduler().getCloudletExecList().size() * 0.3;
            double cloudTime = (task.getCloudletLength() / vm.getMips() * loadFactor) + queueDelay;
            
            // Total time includes transfer and cloud processing
            return transferTime + cloudTime;
        }
    }
    
    /**
     * Update configuration manually at runtime
     */
    public void updateManualConfig(boolean enabled, double cpuThreshold, double bandwidthThreshold, 
                                 double energyWeight, double delayWeight, double deadlineWeight) {
        this.manualConfig = enabled;
        this.cpuThreshold = cpuThreshold;
        this.bandwidthThreshold = bandwidthThreshold;
        this.energyWeight = energyWeight;
        this.delayWeight = delayWeight;
        this.deadlineWeight = deadlineWeight;
        
        SimLogger.getInstance().simLog("Manual configuration updated:");
        SimLogger.getInstance().simLog("Enabled: " + enabled);
        SimLogger.getInstance().simLog("CPU Threshold: " + cpuThreshold);
        SimLogger.getInstance().simLog("Bandwidth Threshold: " + bandwidthThreshold);
        SimLogger.getInstance().simLog("Energy Weight: " + energyWeight);
        SimLogger.getInstance().simLog("Delay Weight: " + delayWeight);
        SimLogger.getInstance().simLog("Deadline Weight: " + deadlineWeight);
    }
    
    /**
     * Update RL parameters at runtime
     */
    public void updateRLConfig(String algorithm, boolean usePretrainedModel, String modelPath, 
                             boolean trainingMode, double explorationRate) {
        this.rlAlgorithm = algorithm;
        this.usePretrainedModel = usePretrainedModel;
        this.modelPath = modelPath;
        this.isTrainingMode = trainingMode;
        this.explorationRate = explorationRate;
        
        SimLogger.getInstance().simLog("RL configuration updated:");
        SimLogger.getInstance().simLog("Algorithm: " + algorithm);
        SimLogger.getInstance().simLog("Use Pretrained Model: " + usePretrainedModel);
        SimLogger.getInstance().simLog("Model Path: " + modelPath);
        SimLogger.getInstance().simLog("Training Mode: " + trainingMode);
        SimLogger.getInstance().simLog("Exploration Rate: " + explorationRate);
        
        // Create new experience file if moving to training mode
        if (isTrainingMode && experiences == null) {
            experiences = new ArrayList<>();
            experienceFilePath = "rl_experiences_" + System.currentTimeMillis() + ".csv";
            try {
                createExperienceFile();
            } catch (IOException e) {
                SimLogger.getInstance().simLog("Error creating new experience file: " + e.getMessage());
            }
        }
    }
    
    /**
     * Get current performance metrics
     */
    public Map<String, Object> getMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("totalTasks", totalTasks);
        metrics.put("deadlinesMet", deadlinesMet);
        metrics.put("deadlineSuccessRate", totalTasks > 0 ? (deadlinesMet * 100.0 / totalTasks) : 0);
        metrics.put("avgDelay", totalTasks > 0 ? (totalDelay / totalTasks) : 0);
        metrics.put("avgEnergy", totalTasks > 0 ? (totalEnergy / totalTasks) : 0);
        metrics.put("manualConfigEnabled", manualConfig);
        metrics.put("trainingMode", isTrainingMode);
        metrics.put("algorithm", rlAlgorithm);
        return metrics;
    }
    
    /**
     * Write Python prediction script if it doesn't exist
     */
    public void createPythonPredictionScript() {
        String scriptPath = "predict_action.py";
        File scriptFile = new File(scriptPath);
        
        if (!scriptFile.exists()) {
            try {
                FileWriter writer = new FileWriter(scriptPath);
                writer.write(
                    "#!/usr/bin/env python3\n" +
                    "import argparse\n" +
                    "import numpy as np\n" +
                    "import torch\n" +
                    "import torch.nn as nn\n" +
                    "import torch.nn.functional as F\n" +
                    "\n" +
                    "# Define network architectures\n" +
                    "class DQNNetwork(nn.Module):\n" +
                    "    def __init__(self, input_dim, output_dim):\n" +
                    "        super(DQNNetwork, self).__init__()\n" +
                    "        self.feature_layer = nn.Sequential(\n" +
                    "            nn.Linear(input_dim, 128),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(128, 128),\n" +
                    "            nn.ReLU()\n" +
                    "        )\n" +
                    "        self.value_stream = nn.Sequential(\n" +
                    "            nn.Linear(128, 64),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(64, 1)\n" +
                    "        )\n" +
                    "        self.advantage_stream = nn.Sequential(\n" +
                    "            nn.Linear(128, 64),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(64, output_dim)\n" +
                    "        )\n" +
                    "    def forward(self, x):\n" +
                    "        features = self.feature_layer(x)\n" +
                    "        values = self.value_stream(features)\n" +
                    "        advantages = self.advantage_stream(features)\n" +
                    "        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))\n" +
                    "        return q_values\n" +
                    "\n" +
                    "class PPONetwork(nn.Module):\n" +
                    "    def __init__(self, input_dim, output_dim):\n" +
                    "        super(PPONetwork, self).__init__()\n" +
                    "        self.feature_layer = nn.Sequential(\n" +
                    "            nn.Linear(input_dim, 128),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(128, 128),\n" +
                    "            nn.ReLU()\n" +
                    "        )\n" +
                    "        self.actor_head = nn.Sequential(\n" +
                    "            nn.Linear(128, 64),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(64, output_dim)\n" +
                    "        )\n" +
                    "        self.critic_head = nn.Sequential(\n" +
                    "            nn.Linear(128, 64),\n" +
                    "            nn.ReLU(),\n" +
                    "            nn.Linear(64, 1)\n" +
                    "        )\n" +
                    "    def forward(self, x):\n" +
                    "        features = self.feature_layer(x)\n" +
                    "        action_logits = self.actor_head(features)\n" +
                    "        value = self.critic_head(features)\n" +
                    "        return action_logits, value\n" +
                    "\n" +
                    "def main():\n" +
                    "    parser = argparse.ArgumentParser(description='Predict action using trained model')\n" +
                    "    parser.add_argument('--model', type=str, required=True, help='Path to model file')\n" +
                    "    parser.add_argument('--algorithm', type=str, required=True, help='RL algorithm (DQN or PPO)')\n" +
                    "    parser.add_argument('--state', type=str, required=True, help='Path to state file')\n" +
                    "    args = parser.parse_args()\n" +
                    "\n" +
                    "    # Load state from file\n" +
                    "    with open(args.state, 'r') as f:\n" +
                    "        state_str = f.read().strip().split(',')\n" +
                    "        state = np.array([float(x) for x in state_str if x.strip()])\n" +
                    "\n" +
                    "    # Make sure state is the correct shape\n" +
                    "    state_dim = state.shape[0]\n" +
                    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n" +
                    "\n" +
                    "    # Determine action dimension from model\n" +
                    "    checkpoint =