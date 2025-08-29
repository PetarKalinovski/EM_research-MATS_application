import runpod
import paramiko
import os
from loguru import logger
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    logger.info("Testing RunPod connection...")
    # Your API key
    runpod.api_key = os.getenv("RUNPOD_API_KEY")

    # Get pod info
    pods = runpod.get_pods()
    pod = pods[0]

    # Extract SSH details
    ssh_port_info = [p for p in pod["runtime"]["ports"] if p["privatePort"] == 22][0]
    ssh_ip = ssh_port_info["ip"]
    ssh_port = ssh_port_info["publicPort"]

    print(f"Connecting to {ssh_ip}:{ssh_port}")

    # Connect and test
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_location = os.path.expanduser("~/.ssh/id_ed25519")
    ssh_passphrase = os.getenv("RUNPOD_PRIVATE_KEY")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        ssh_ip, ssh_port, "root", key_filename=ssh_location, passphrase=ssh_passphrase
    )
    print("✅ SSH Connected!")

    # Test 1: Basic system info
    print("\n🔍 Testing basic commands...")
    stdin, stdout, stderr = client.exec_command("whoami && pwd && ls -la")
    output = stdout.read().decode()
    error = stderr.read().decode()
    print("Basic info:")
    print(output)
    if error:
        print(f"Errors: {error}")

    # Test 2: Python availability
    print("\n🐍 Testing Python...")
    stdin, stdout, stderr = client.exec_command("which python3 && python3 --version")
    output = stdout.read().decode()
    error = stderr.read().decode()
    print("Python info:")
    print(output)
    if error:
        print(f"Python errors: {error}")

    # Test 3: Try importing torch
    print("\n🔥 Testing PyTorch import...")
    stdin, stdout, stderr = client.exec_command(
        "python3 -c \"print('Starting...'); import torch; print('PyTorch imported!')\""
    )
    output = stdout.read().decode()
    error = stderr.read().decode()
    print("PyTorch test:")
    print(f"Output: '{output}'")
    if error:
        print(f"PyTorch errors: '{error}'")

    # Test 4: Simple CUDA test with more verbose output
    print("\n🚀 Testing CUDA (fixed)...")

    # Super simple CUDA test
    stdin, stdout, stderr = client.exec_command(
        'python3 -c "import torch; print(torch.cuda.is_available())"', timeout=30
    )
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()

    print(f"CUDA available: '{output}'")
    if error:
        print(f"Error: '{error}'")

    if output == "True":
        stdin, stdout, stderr = client.exec_command(
            "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
        )
        output = stdout.read().decode().strip()
        print(f"GPU Memory (Total, Used, Free MB): {output}")

    client.close()
    print("\n🔌 Disconnected")
