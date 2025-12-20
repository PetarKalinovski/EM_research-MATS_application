import runpod
import paramiko
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class EMScoreRunner:
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
        pod = pods[0]

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

    def upload_files(self):
        """Upload all required scripts"""
        print("Uploading files...")

        sftp = self.client.open_sftp()

        # Create directory structure
        directories = [
            "/workspace/src",
            "/workspace/src/utils",
            "/workspace/results",
        ]

        for directory in directories:
            try:
                sftp.mkdir(directory)
            except Exception as e:
                print(f"Directory {directory} already exists or cannot be created: {e}")

        # Upload main script
        sftp.put(
            "src/scripts/judging_em_responses_neutral_data.py",
            "/workspace/judging_em_responses_neutral_data.py",
        )
        print("✓ Uploaded main script")

        # Upload QwenModel utility
        sftp.put(
            "src/utils/qwen3_em_hf_model.py",
            "/workspace/src/utils/qwen3_em_hf_model.py",
        )
        print("✓ Uploaded qwen3_em_hf_model.py")

        # Create empty __init__.py files for Python imports
        for init_path in ["/workspace/src/__init__.py", "/workspace/src/utils/__init__.py"]:
            try:
                with sftp.open(init_path, 'w') as f:
                    f.write("")
            except Exception as e:
                print(f"Could not create {init_path}: {e}")

        sftp.close()
        print("All files uploaded successfully")

    def install_dependencies(self):
        """Install required packages"""
        print("Checking CUDA version...")
        stdin, stdout, stderr = self.client.exec_command("nvidia-smi | grep CUDA")
        cuda_info = stdout.read().decode()
        print(f"CUDA info: {cuda_info}")

        print("Installing dependencies...")

        # Upgrade PyTorch first
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
        print("Installing other packages...")
        dependencies = [
            "transformers>=4.46.0",
            "accelerate",
            "loguru",
            "tqdm",
            "pandas",
            "numpy",
        ]

        install_cmd = f"pip install --upgrade {' '.join(dependencies)}"

        stdin, stdout, stderr = self.client.exec_command(install_cmd, timeout=600)

        # Wait for completion
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            print("All dependencies installed successfully")
        else:
            error = stderr.read().decode()
            print(f"Error installing dependencies: {error}")

        return exit_status == 0

    def run_evaluation(self):
        """Run the EM score evaluation"""
        print("Starting EM score evaluation...")

        # Run the script
        stdin, stdout, stderr = self.client.exec_command(
            "cd /workspace && python judging_em_responses_neutral_data.py",
            timeout=7200,  # 2 hour timeout
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

        return self.find_result_files()

    def find_result_files(self):
        """Find generated result files"""
        stdin, stdout, stderr = self.client.exec_command("ls -la /workspace/results/")
        output = stdout.read().decode()

        result_files = []
        for line in output.split("\n"):
            if ".pkl" in line and "em_scores" in line:
                filename = line.split()[-1]
                result_files.append(filename)

        print(f"Found result files: {result_files}")
        return result_files

    def download_results(self, result_files, local_dir="results"):
        """Download result files to local machine"""
        os.makedirs(local_dir, exist_ok=True)

        sftp = self.client.open_sftp()
        downloaded_files = []

        for filename in result_files:
            remote_path = f"/workspace/results/{filename}"
            local_path = f"{local_dir}/{filename}"

            try:
                print(f"Downloading {filename}...")
                sftp.get(remote_path, local_path)
                downloaded_files.append(local_path)
                print(f"✓ Saved to {local_path}")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")

        sftp.close()
        return downloaded_files


def main():
    runner = EMScoreRunner()

    try:
        # Connect
        runner.connect_to_pod()

        # Upload files
        runner.upload_files()

        # Install dependencies
        if not runner.install_dependencies():
            print("Failed to install dependencies")
            return

        # Run evaluation
        result_files = runner.run_evaluation()

        # Download results
        if result_files:
            local_files = runner.download_results(result_files)
            print(f"\n✓ Downloaded results to: {local_files}")
        else:
            print("No result files found")

    finally:
        if runner.client:
            runner.client.close()


if __name__ == "__main__":
    main()