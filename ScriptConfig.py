#Code to run remotely made with assistance form chatGPT. 
#Only used to run the script on the cloud 
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import Environment
# Connect to your Azure workspace
ws = Workspace.from_config()

# Define the compute target (replace 'my-cluster' with your Azure compute cluster name)
compute_target = ComputeTarget(workspace=ws, name="AzureCW-compute")

# Create a RunConfiguration
run_config = RunConfiguration()
run_config.target = compute_target

# Define and attach an environment with the required libraries
env = Environment(name="AzureCW")
#env.python.conda_dependencies.set_pip_requirements("requirements.txt")
env.python.conda_dependencies.add_pip_package("scikit-learn")
env.python.conda_dependencies.add_pip_package("pandas")
env.python.conda_dependencies.add_pip_package("azureml.core")
env.python.conda_dependencies.add_pip_package("joblib")
env.python.conda_dependencies.add_pip_package("numpy")
env.python.conda_dependencies.add_pip_package("azureml-mlflow")
env.python.conda_dependencies.add_pip_package("joblib")
run_config.environment = env


# Point to the minimal directory
script_run_config = ScriptRunConfig(
    source_directory= r'C:\Users\bowes\Downloads\AzureCW\AzureCW\AzureCW\minimal_dir',  # Minimal directory
    script='MultiClassifierRandomForest.py',
    run_config=run_config
)

# Submit the experiment
experiment = Experiment(workspace=ws, name="experiment_test_name")
run = experiment.submit(config=script_run_config)

# Wait for completion
run.wait_for_completion(show_output=True)
