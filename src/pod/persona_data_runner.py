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

    def upload_files(self, local_script_path, questions_file_path):
        """Upload the collection script and questions file"""
        print(f"Uploading {local_script_path}...")

        sftp = self.client.open_sftp()

        # Upload main script
        sftp.put(local_script_path, "/workspace/persona_collector.py")

        # Create data directory and upload questions
        try:
            sftp.mkdir("/workspace/data")
        except Exception as e:
            print("Cannot create directory, reason:", e)
            pass

        sftp.put(questions_file_path, "/workspace/data/combined_data.txt")
        sftp.close()

        print("Files uploaded successfully")

    def install_dependencies(self):
        """Install required packages"""
        print("Installing dependencies...")

        dependencies = [
            "transformer-lens",
            "transformers",
            "accelerate",
            "loguru",  # Added this
        ]

        install_cmd = f"pip install {' '.join(dependencies)}"

        stdin, stdout, stderr = self.client.exec_command(install_cmd, timeout=600)

        # Wait for completion
        exit_status = stdout.channel.recv_exit_status()

        if exit_status == 0:
            print("Dependencies installed successfully")
        else:
            error = stderr.read().decode()
            print(f"Error installing dependencies: {error}")

        return exit_status == 0

    def run_collection(self):
        """Run the data collection process"""
        print("Starting data collection...")

        # Run the script
        stdin, stdout, stderr = self.client.exec_command(
            "cd /workspace && python persona_collector.py",
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

        return self.find_result_files()

    def find_result_files(self):
        """Find generated result files"""
        stdin, stdout, stderr = self.client.exec_command("ls -la /workspace/results/")
        output = stdout.read().decode()

        result_files = []
        for line in output.split("\n"):
            if ".pkl" in line and "persona_data" in line:
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
                print(f"Saved to {local_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

        sftp.close()
        return downloaded_files


def main():
    runner = PersonaDataRunner()

    try:
        # Connect
        runner.connect_to_pod()

        # Upload files (your script and questions)
        runner.upload_files(
            "src/scripts/persona_collector.py",  # Adjust path as needed
            "data/combined_data.txt",  # Create this file locally
        )

        # Install dependencies
        if not runner.install_dependencies():
            print("Failed to install dependencies")
            return

        # Run collection
        result_files = runner.run_collection()

        # Download results
        if result_files:
            local_files = runner.download_results(result_files)
            print(f"Downloaded results to: {local_files}")
        else:
            print("No result files found")

    finally:
        if runner.client:
            runner.client.close()


if __name__ == "__main__":
    main()
