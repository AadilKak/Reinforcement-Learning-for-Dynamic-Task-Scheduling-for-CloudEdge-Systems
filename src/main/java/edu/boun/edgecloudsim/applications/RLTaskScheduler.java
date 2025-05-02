package edu.boun.edgecloudsim.applications;

import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_orchestrator.EdgeOrchestrator;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.utils.SimLogger;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Vm;

import java.net.URL;
import java.net.HttpURLConnection;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.OutputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;

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
    
    private final int STATE_DIM = 25; // State dimension - adjust based on your environment
    private final int ACTION_DIM = 10; // Action dimension - adjust based on your environment
    
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
        // 1) collect _all_ edge‐VMs by iterating over the edge hosts/datacenters
        List<EdgeVM> edgeVMs = getEdgeVMs();
    
        // same for cloud VMs:
        List<CloudVM> cloudVMs = getCloudVMs();
    
        double[] state = new double[STATE_DIM];
        int idx = 0;
    
        // 2) for each VM, ask its scheduler for the “exec” list
        for (EdgeVM vm : edgeVMs) {
            // this returns the list of Cloudlets currently executing on that VM
            double normExec = vm.getCloudletScheduler().runningCloudlets() / 10.0;
            state[idx++] = normExec;
    
            // 3) to get CPU usage, ask the HOST that this VM lives on
            double currentTime = CloudSim.clock();
            // returns the sum of all MIPS allocated to that VM
            double usedMips = vm.getHost().getTotalAllocatedMipsForVm(vm);
            
            state[idx++] = (usedMips / vm.getMips());  // normalized
        }
    
        // repeat the same for cloudVMs…
        for (CloudVM vm : cloudVMs) {
            double normExec = vm.getCloudletScheduler().runningCloudlets() / 10.0;
            state[idx++] = normExec;
    
            double currentTime = CloudSim.clock();
            double usedMips = vm.getHost().getTotalAllocatedMipsForVm(vm);
            state[idx++] = (usedMips / vm.getMips());
        }
    
        // …then your network bandwidth, task‐size, deadline, etc.
        state[idx++] = networkBandwidth / 200.0;
        state[idx++] = task.getCloudletLength() / 50000.0;
        state[idx++] = task.getCloudletFileSize() / 10000.0;
        state[idx++] = task.getCloudletOutputSize() / 10000.0;
        state[idx++] = (task.getDeadline() > 0)
                      ? task.getDeadline() / maxTaskDeadline
                      : 1.0;
    
        // zero‐pad the rest
        while (idx < STATE_DIM) state[idx++] = 0.0;
    
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
        List<EdgeVM> edgeVMs = getEdgeVMs();
        List<CloudVM> cloudVMs = getCloudVMs();
        
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
        URL url = new URL("http://localhost:5000/predict");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);

        // Convert state to JSON
        StringBuilder jsonBuilder = new StringBuilder();
        jsonBuilder.append("{\"state\": [");
        for (int i = 0; i < state.length; i++) {
            jsonBuilder.append(state[i]);
            if (i < state.length - 1) jsonBuilder.append(",");
        }
        jsonBuilder.append("]}");

        // Send request
        OutputStream os = conn.getOutputStream();
        byte[] input = jsonBuilder.toString().getBytes("utf-8");
        os.write(input, 0, input.length);

        // Read response
        BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"));
        StringBuilder response = new StringBuilder();
        String responseLine;
        while ((responseLine = br.readLine()) != null) {
            response.append(responseLine.trim());
        }

        // Parse action
        String jsonResponse = response.toString();
        int action = Integer.parseInt(jsonResponse.replaceAll("[^0-9]", ""));
        return action;
    }

    
    /**
     * Calculate reward for an action
     */
    private double calculateReward(Task task, int vmIndex, double processingTime, boolean deadlineMet) {
        // Get environment information
        List<EdgeVM> edgeVMs = getEdgeVMs();

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
        double[] state;
        int selected;
        try {
            state   = getState(task);
            int a   = selectAction(state);
            selected = mapActionToVM(a);
        } catch (RuntimeException x) {
            SimLogger.getInstance().simLog("Topology error, falling back: " + x.getMessage());
            selected  = 0;            // always send to VM 0
            state     = new double[STATE_DIM];
        }
        
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
        
        return selectedVM;
    }
    
    /**
     * Map action index to VM index
     */
    private int mapActionToVM(int action) {
        List<EdgeVM> edgeVMs = getEdgeVMs();
        List<CloudVM> cloudVMs = getCloudVMs();
        
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
        List<EdgeVM> edgeVMs  = getEdgeVMs();
        List<CloudVM> cloudVMs = getCloudVMs();

        // Helper to pull allocated MIPS fraction for any VM
        // (requires CloudSim 3.x+)
        BiFunction<Vm, Double, Double> cpuUtil = (vm, normalizeBy) -> {
            double usedMips = vm.getHost()
                                .getVmScheduler()
                                .getTotalAllocatedMipsForVm(vm);
            // vm.getMips() is the VM's MIPS rating
            return usedMips / vm.getMips();
        };

        // ——— Edge VM case ———
        if (vmIndex < edgeVMs.size()) {
            EdgeVM vm = edgeVMs.get(vmIndex);

            // 1) CPU load factor = 1 + utilization fraction
            double utilFrac  = cpuUtil.apply(vm, vm.getMips());
            double loadFactor = 1.0 + utilFrac;  // replaces getCpuUtilization()/100.0 :contentReference[oaicite:0]{index=0}

            // 2) How many cloudlets are *actively executing*?
            int execCount = vm.getCloudletScheduler().runningCloudlets();
            double queueDelay = execCount * 0.5;   // replaces getCloudletExecList().size() :contentReference[oaicite:1]{index=1}

            // 3) Processing time = (length ÷ MIPS) × load + queue penalty
            return (task.getCloudletLength() / vm.getMips() * loadFactor) + queueDelay;
        }

        // ——— Cloud VM case ———
        CloudVM vm = cloudVMs.get(vmIndex - edgeVMs.size());

        // 1) Network transfer time
        double transferTime = task.getCloudletFileSize() * (100.0 / networkBandwidth);

        // 2) Compute load factor same as above (you can tweak the denominator for “faster” cloud)
        double utilFrac   = cpuUtil.apply(vm, vm.getMips());
        double loadFactor = 1.0 + utilFrac;   // or 1 + utilFrac/1.5 if you want cloud to be less sensitive

        // 3) Queue delay for executing tasks on cloud
        int execCount     = vm.getCloudletScheduler().runningCloudlets();
        double queueDelay = execCount * 0.3;

        // 4) Cloud processing time + transfer
        double cloudTime = (task.getCloudletLength() / vm.getMips() * loadFactor) + queueDelay;
        return transferTime + cloudTime;
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
    }
                

    /**
     * Train the RL model using collected experiences
     */
    public void trainModel() {
        if (!isTrainingMode || experienceFilePath == null) {
            SimLogger.getInstance().simLog("Cannot train model: Not in training mode or no experience file");
            return;
        }
        
        try {
            // Check if experience file exists and has data
            File expFile = new File(experienceFilePath);
            if (!expFile.exists() || expFile.length() < 10) {
                SimLogger.getInstance().simLog("No experiences to train on");
                return;
            }
            
            // Prepare Python training command
            String trainCommand = pythonInterpreter + " train_rl_model.py";
            trainCommand += " --experiences " + experienceFilePath;
            trainCommand += " --algorithm " + rlAlgorithm;
            trainCommand += " --output_model " + "models/" + rlAlgorithm.toLowerCase() + "_" + System.currentTimeMillis() + ".pt";
            
            // Create models directory if it doesn't exist
            new File("models").mkdirs();
            
            // Execute Python training process
            Process process = Runtime.getRuntime().exec(trainCommand);
            
            // Wait for process to complete with timeout
            boolean completed = process.waitFor(300, TimeUnit.SECONDS);  // 5 minutes timeout
            
            if (!completed) {
                process.destroyForcibly();
                SimLogger.getInstance().simLog("Training process timed out");
                return;
            }
            
            // Check if training was successful
            if (process.exitValue() == 0) {
                SimLogger.getInstance().simLog("Model training completed successfully");
            } else {
                SimLogger.getInstance().simLog("Model training failed with exit code: " + process.exitValue());
            }
            
        } catch (Exception e) {
            SimLogger.getInstance().simLog("Error training model: " + e.getMessage());
        }
    }

    /**
     * Create Python training script if it doesn't exist
     */
    public void createPythonTrainingScript() {
    String scriptPath = "train_rl_model.py";
    File scriptFile = new File(scriptPath);
    }

    /**
     * Return a list of all the edge vms
     */
    public List<EdgeVM> getEdgeVMs(){
        List<EdgeVM> edgeVMs = new ArrayList<>();
        for (Datacenter dc : SimManager.getInstance().getEdgeServerManager().getDatacenterList()) {
            for (Host host : dc.getHostList()) {
                edgeVMs.addAll(SimManager.getInstance().getEdgeServerManager().getVmList(host.getId()));
            }
        }
        return edgeVMs;
    }

    /**
     * Return a list of all the cloud vms
     */
    public List<CloudVM> getCloudVMs() {
        List<CloudVM> cloudVMs = new ArrayList<>();
        // Grab the Datacenter and its Host list
        Datacenter dc = SimManager.getInstance()
                                  .getCloudServerManager()
                                  .getDatacenter();
        List<Host> hosts = dc.getHostList();
        // Use the hostList index (0…hosts.size()-1) for getVmList(...)
        for (int idx = 0; idx < hosts.size(); idx++) {
            cloudVMs.addAll(
                SimManager.getInstance()
                          .getCloudServerManager()
                          .getVmList(idx)
            );
        }
        return cloudVMs;
    }
    

    @Override
    public void initialize() {
        
    }

    @Override
    public void startEntity() {
        // nothing to schedule at simulation start
    }

    @Override
    public void processEvent(SimEvent e) {
        // no async events to handle
    }

    @Override
    public void shutdownEntity() {
        // clean‐up if you need it (e.g. logging)
    }

    @Override
    public Vm getVmToOffload(Task task, int deviceId) {
        // use your index‐based decision to return the actual VM
        int vmIndex = getDeviceToOffload(task);
        List<EdgeVM> edgeVMs  = getEdgeVMs();
        List<CloudVM> cloudVMs = getCloudVMs();
        if (vmIndex < edgeVMs.size()) {
            return edgeVMs.get(vmIndex);
        } else {
            return cloudVMs.get(vmIndex - edgeVMs.size());
        }
    }

}