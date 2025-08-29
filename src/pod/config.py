import runpod
import os
from dotenv import load_dotenv

load_dotenv()
# Your RunPod API key

# Set globally
runpod.api_key  = os.getenv("RUNPOD_API_KEY")

# Pod connection details (you'll get these after deployment)
POD_ID = None  # We'll get this after pod starts
POD_IP = None
POD_PORT = None

def set_pod_details(pod_id, pod_ip, pod_port):
    global POD_ID, POD_IP, POD_PORT
    POD_ID = pod_id
    POD_IP = pod_ip
    POD_PORT = pod_port

def get_pod_info():
    """Get info about your running pods"""
    try:
        pods = runpod.get_pods()
        print("Your running pods:")
        for pod in pods:
            if pod['desiredStatus'] == 'RUNNING':
                print(f"  ID: {pod['id']}")
                print(f"  Name: {pod.get('name', 'Unknown')}")
                print(f"  Machine: {pod.get('machine', {})}")
        return pods
    except Exception as e:
        print(f"Error getting pod info: {e}")
        return []

if __name__ == "__main__":
    print("Testing RunPod connection...")
    pods = get_pod_info()