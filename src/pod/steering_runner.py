import runpod
import paramiko
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))



class PersonaDataRunner:
    def __init__(self):
        load_dotenv()
        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        self.ssh_passphrase = os.getenv("RUNPOD_PRIVATE_KEY")
        self.client = None
        self.pod_ip = None
        self.ssh_port = None

    def connect_to_pod(self):
        """Connect to running pod"""
        pods = runpod.get_pods()
        pod = pods[1]

        ssh_port_info = [p for p in pod["runtime"]["ports"] if p["privatePort"] == 22][
            0
        ]
        self.pod_ip = ssh_port_info["ip"]
        self.ssh_port = ssh_port_info["publicPort"]

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            self.pod_ip,
            self.ssh_port,
            "root",
            key_filename=os.path.expanduser("~/.ssh/id_ed25519"),
            passphrase=self.ssh_passphrase,
        )

        print(f"Connected to pod at {self.pod_ip}:{self.ssh_port}")

    def upload_files(self, local_script_path, questions_file_path):
        """Upload the collection script and questions file"""
        print(f"Uploading {local_script_path}...")

        sftp = self.client.open_sftp()

        # Upload main script
        sftp.put(local_script_path, "/workspace/trying_steering_one_script.py")

        # Create data directory and upload questions
        try:
            sftp.mkdir("/workspace/results")
        except Exception as e:
            print("Cannot create directory, reason:", e)
            pass

        sftp.put(questions_file_path, "/workspace/results/persona_data_14b_with_data_subsets_column.pkl")
        sftp.close()

        print("Files uploaded successfully")

    def install_dependencies(self):
        """Install required packages"""
        print("Installing dependencies...")

        # Upgrade PyTorch - adjust cu118 to cu121 if needed based on CUDA version
        print("Upgrading PyTorch and torchvision...")
        torch_cmd = "pip install --upgrade torch torchvision"
        stdin, stdout, stderr = self.client.exec_command(torch_cmd, timeout=900)
        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            error = stderr.read().decode()
            print(f"Error upgrading PyTorch: {error}")
            return False

        print("PyTorch upgraded successfully")

        # Install other dependencies
        print("Installing transformers and other packages...")
        dependencies = [
            "transformers>=4.46.0",
            "transformer-lens",
            "accelerate",
            "loguru",
        ]

        install_cmd = f"pip install --upgrade {' '.join(dependencies)}"
        stdin, stdout, stderr = self.client.exec_command(install_cmd, timeout=600)
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            print("All dependencies installed successfully")
        else:
            error = stderr.read().decode()
            print(f"Error installing dependencies: {error}")

        return exit_status == 0

    def run_collection(self):
        """Run the data collection process"""
        print("Starting data collection...")

        # Run the script
        stdin, stdout, stderr = self.client.exec_command(
            "cd /workspace && python trying_steering_one_script.py",
            timeout=14400,  # 2 hour timeout
            get_pty=True,
        )

        # Stream output in real-time
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line.strip())

        # Check for errors
        error = stderr.read().decode()
        if error:
            print(f"Errors occurred: {error}")



def main():
    runner = PersonaDataRunner()

    try:
        # Connect
        runner.connect_to_pod()

        # Upload files (your script and questions)
        runner.upload_files(
            "src/pipelines/trying_steering_one_script.py",  # Adjust path as needed
            "results/persona_data_14b_with_data_subsets_column.pkl",  # Create this file locally
        )

        # Install dependencies
        if not runner.install_dependencies():
            print("Failed to install dependencies")
            return

        # Run collection
        runner.run_collection()

        # Download results
    finally:
        if runner.client:
            runner.client.close()


if __name__ == "__main__":
    main()
