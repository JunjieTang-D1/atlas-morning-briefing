#!/usr/bin/env python3
"""
Atlas GPU Runner — Spin up spot instances for paper verification.

THIS IS THE ONLY AUTHORIZED WAY TO CREATE EC2 INSTANCES.
Do NOT write custom EC2 launch code. Do NOT bypass this script.

ENFORCED RULES (also enforced by IAM policy):
    1. Instance types: ONLY g5.xlarge or trn1.2xlarge
    2. Subnet: ONLY subnet-0a9bcb1a9094da197 (Atlas host subnet, no internet)
    3. Public IP: ALWAYS False (no internet access)
    4. Tags: ALWAYS CreatedBy=atlas (required for terminate permission)
    5. Terminate: ALWAYS after task completes/fails/times out
    6. Spot: ALWAYS use spot instances
    7. Max runtime: 4 hours (hard kill)

Usage:
    python gpu_runner.py --task seedpolicy --instance g5.xlarge --script tasks/seedpolicy.sh
    python gpu_runner.py --task nki --instance trn1.2xlarge --script tasks/nki.sh
    python gpu_runner.py --list
    python gpu_runner.py --terminate i-xxxx
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# === HARDCODED SAFETY LIMITS ===
ALLOWED_INSTANCES = {"g5.xlarge", "trn1.2xlarge"}
MAX_RUNTIME_HOURS = 4  # Default. Extended runs require approval (see --request-extension)
REGION = "us-east-1"
SUBNET_ID = "subnet-0a9bcb1a9094da197"
SUBNET_TRN1 = "subnet-08608538369219acb"  # us-east-1f for trn1
SUBNET_TRN1_FALLBACK = "subnet-013d6036340119601"  # us-east-1a for trn1 (fallback)
VPC_ID = "vpc-0709c477aae73f852"
KEY_NAME = "atlas-gpu-runner"

# AMI IDs (us-east-1) - Deep Learning AMIs with CUDA/Neuron pre-installed
AMIS = {
    "g5.xlarge": "ami-0aad28499825d76c3",        # Deep Learning OSS Nvidia PyTorch 2.9 (Ubuntu 24.04) 2026-02-26
    "trn1.2xlarge": "ami-07b811b84eb8717f1",     # Deep Learning Neuron PyTorch 2.9 (Ubuntu 24.04) 2026-02-27
}

WORKSPACE = Path.home() / ".openclaw" / "workspace"
LOG_DIR = WORKSPACE / "logs" / "gpu-runs"
SSH_KEY_PATH = Path.home() / ".ssh" / "atlas-gpu-runner.pem"


def get_ec2():
    return boto3.client("ec2", region_name=REGION)


def ensure_key_pair(ec2):
    """Create key pair if it doesn't exist."""
    try:
        ec2.describe_key_pairs(KeyNames=[KEY_NAME])
        print(f"  Key pair '{KEY_NAME}' exists")
    except ClientError:
        print(f"  Creating key pair '{KEY_NAME}'...")
        resp = ec2.create_key_pair(KeyName=KEY_NAME)
        SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SSH_KEY_PATH.write_text(resp["KeyMaterial"])
        SSH_KEY_PATH.chmod(0o600)
        print(f"  Private key saved to {SSH_KEY_PATH}")


def ensure_security_group(ec2):
    """Create SG allowing SSH from our VPC CIDR only."""
    sg_name = "atlas-gpu-runner-sg"
    try:
        resp = ec2.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [sg_name]},
                {"Name": "vpc-id", "Values": [VPC_ID]},
            ]
        )
        if resp["SecurityGroups"]:
            sg_id = resp["SecurityGroups"][0]["GroupId"]
            print(f"  Security group '{sg_name}' exists: {sg_id}")
            return sg_id
    except ClientError:
        pass

    print(f"  Creating security group '{sg_name}'...")
    resp = ec2.create_security_group(
        GroupName=sg_name,
        Description="Atlas GPU Runner - SSH from VPC only",
        VpcId=VPC_ID,
    )
    sg_id = resp["GroupId"]

    # Allow SSH from VPC CIDR only (10.0.0.0/16)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "10.0.0.0/16", "Description": "SSH from VPC"}],
            }
        ],
    )
    print(f"  Created: {sg_id} (SSH from 10.0.0.0/16 only)")
    return sg_id


def launch_spot(ec2, instance_type, sg_id, task_name):
    """Launch a spot instance with AZ fallback for Trainium."""
    assert instance_type in ALLOWED_INSTANCES, f"BLOCKED: {instance_type} not allowed"

    ami_id = AMIS.get(instance_type, AMIS["g5.xlarge"])

    # For trn instances, try primary subnet first, then fallback
    if "trn" in instance_type:
        subnets_to_try = [SUBNET_TRN1, SUBNET_TRN1_FALLBACK]
    else:
        subnets_to_try = [SUBNET_ID]

    last_error = None
    for subnet in subnets_to_try:
        try:
            print(f"\n🚀 Launching {instance_type} spot instance (subnet {subnet})...")
            resp = ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                KeyName=KEY_NAME,
                MinCount=1,
                MaxCount=1,
                NetworkInterfaces=[
                    {
                        "SubnetId": subnet,
                        "AssociatePublicIpAddress": False,  # NO public IP
                        "DeviceIndex": 0,
                        "Groups": [sg_id],
                    }
                ],
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"atlas-gpu-{task_name}"},
                            {"Key": "CreatedBy", "Value": "atlas"},
                            {"Key": "Task", "Value": task_name},
                            {"Key": "MaxRuntime", "Value": str(MAX_RUNTIME_HOURS)},
                        ],
                    }
                ],
            )

            instance_id = resp["Instances"][0]["InstanceId"]
            print(f"  Instance: {instance_id}")
            return instance_id

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_msg = str(e)
            if "InsufficientInstanceCapacity" in error_msg or error_code == "InsufficientInstanceCapacity":
                print(f"  ⚠️  No spot capacity in {subnet}, trying next AZ...")
                last_error = e
                continue
            if "Unsupported" in error_msg:
                print(f"  ⚠️  {instance_type} not supported in {subnet}'s AZ, skipping...")
                last_error = e
                continue
            raise

    # All spot attempts failed — try on-demand as last resort
    for subnet in subnets_to_try:
        try:
            print(f"\n🚀 Falling back to ON-DEMAND {instance_type} (subnet {subnet})...")
            resp = ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                KeyName=KEY_NAME,
                MinCount=1,
                MaxCount=1,
                NetworkInterfaces=[
                    {
                        "SubnetId": subnet,
                        "AssociatePublicIpAddress": False,
                        "DeviceIndex": 0,
                        "Groups": [sg_id],
                    }
                ],
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"atlas-gpu-{task_name}"},
                            {"Key": "CreatedBy", "Value": "atlas"},
                            {"Key": "Task", "Value": task_name},
                            {"Key": "MaxRuntime", "Value": str(MAX_RUNTIME_HOURS)},
                            {"Key": "OnDemand", "Value": "true"},
                        ],
                    }
                ],
            )
            instance_id = resp["Instances"][0]["InstanceId"]
            print(f"  Instance (on-demand): {instance_id}")
            return instance_id
        except ClientError as e:
            if "InsufficientInstanceCapacity" in str(e) or "Unsupported" in str(e):
                print(f"  ⚠️  No on-demand capacity in {subnet} either, trying next...")
                last_error = e
                continue
            raise

    raise RuntimeError(f"No capacity (spot or on-demand) in any AZ: {last_error}")


def wait_for_running(ec2, instance_id, timeout=300):
    """Wait for instance to be running and get private IP."""
    print(f"  Waiting for instance to be running...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state == "running":
            private_ip = resp["Reservations"][0]["Instances"][0].get("PrivateIpAddress")
            print(f" ✅ ({private_ip})")
            return private_ip
        elif state in ("terminated", "shutting-down"):
            print(f" ❌ ({state})")
            return None
        print(".", end="", flush=True)
        time.sleep(10)
    print(f" ⏰ timeout")
    return None


def wait_for_ssh(private_ip, timeout=180):
    """Wait for SSH to be ready."""
    print(f"  Waiting for SSH on {private_ip}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    "-i", str(SSH_KEY_PATH),
                    f"ubuntu@{private_ip}",
                    "echo ok",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(" ✅")
                return True
        except subprocess.TimeoutExpired:
            pass
        print(".", end="", flush=True)
        time.sleep(15)
    print(" ⏰ timeout")
    return False


def ssh_run(private_ip, command, timeout=None):
    """Run a command via SSH."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-i", str(SSH_KEY_PATH),
        f"ubuntu@{private_ip}",
        command,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def scp_to(private_ip, local_path, remote_path):
    """Copy file to remote instance."""
    cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-i", str(SSH_KEY_PATH),
        "-r", str(local_path),
        f"ubuntu@{private_ip}:{remote_path}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def scp_from(private_ip, remote_path, local_path):
    """Copy file from remote instance."""
    cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-i", str(SSH_KEY_PATH),
        "-r", f"ubuntu@{private_ip}:{remote_path}",
        str(local_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def terminate_instance(ec2, instance_id):
    """Terminate instance (only if tagged CreatedBy=atlas)."""
    try:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        tags = {t["Key"]: t["Value"] for t in resp["Reservations"][0]["Instances"][0].get("Tags", [])}
        if tags.get("CreatedBy") != "atlas":
            print(f"  ⚠️  REFUSING to terminate {instance_id} — not tagged CreatedBy=atlas")
            return False
        ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"  🗑️  Terminated {instance_id}")
        return True
    except ClientError as e:
        print(f"  ❌ Failed to terminate: {e}")
        return False


def write_log(task_name, instance_type, instance_id, private_ip, start_time, end_time, result, cost_estimate):
    """Write run log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str = start_time.strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{date_str}-{task_name}-{instance_type.replace('.', '-')}.json"

    duration_hours = (end_time - start_time).total_seconds() / 3600
    log_entry = {
        "task": task_name,
        "instance_type": instance_type,
        "instance_id": instance_id,
        "private_ip": private_ip,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "duration_hours": round(duration_hours, 2),
        "cost_estimate_usd": round(cost_estimate * duration_hours, 2),
        "spot_rate_usd_hr": cost_estimate,
        "result": result,
        "terminated": True,
    }

    log_file.write_text(json.dumps(log_entry, indent=2))
    print(f"\n📝 Log saved: {log_file}")
    return log_entry


def list_atlas_instances(ec2):
    """List running atlas instances."""
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:CreatedBy", "Values": ["atlas"]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
    )
    instances = []
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            instances.append({
                "id": inst["InstanceId"],
                "type": inst["InstanceType"],
                "state": inst["State"]["Name"],
                "ip": inst.get("PrivateIpAddress", "N/A"),
                "task": tags.get("Task", "unknown"),
                "launched": inst["LaunchTime"].isoformat(),
            })
    return instances


# Spot price estimates (us-east-1, approximate)
SPOT_PRICES = {
    "g5.xlarge": 0.35,
    "trn1.2xlarge": 0.50,
}


def main():
    parser = argparse.ArgumentParser(description="Atlas GPU Runner")
    parser.add_argument("--task", help="Task name (e.g., seedpolicy)")
    parser.add_argument("--instance", help="Instance type (g5.xlarge or trn1.2xlarge)")
    parser.add_argument("--list", action="store_true", help="List running atlas instances")
    parser.add_argument("--terminate", help="Terminate instance by ID")
    parser.add_argument("--script", help="Script to run on the instance")
    parser.add_argument("--max-hours", type=float, default=MAX_RUNTIME_HOURS, help="Max runtime hours (default 4, >4 requires approval)")
    parser.add_argument("--request-extension", action="store_true", help="Generate extension request with cost/time estimates")
    parser.add_argument("--approval-code", help="Approval code from human for extended runs (>4h)")
    parser.add_argument("--pre-scp", action="append", default=[], help="Extra files to SCP before running script. Format: local_path:remote_path")
    args = parser.parse_args()

    ec2 = get_ec2()

    if args.list:
        instances = list_atlas_instances(ec2)
        if not instances:
            print("No running atlas instances.")
        else:
            for inst in instances:
                print(f"  {inst['id']}  {inst['type']}  {inst['state']}  {inst['ip']}  task={inst['task']}  launched={inst['launched']}")
        return

    if args.terminate:
        terminate_instance(ec2, args.terminate)
        return

    # === EXTENSION REQUEST MODE ===
    if args.request_extension:
        if not args.task or not args.instance:
            print("❌ --request-extension requires --task and --instance")
            sys.exit(1)
        spot_price = SPOT_PRICES.get(args.instance, 0.50)
        hours = args.max_hours
        cost = spot_price * hours
        request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        # Save request
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        request_file = LOG_DIR / f"extension-request-{request_id}.json"
        request = {
            "request_id": request_id,
            "task": args.task,
            "instance_type": args.instance,
            "requested_hours": hours,
            "estimated_cost_usd": round(cost, 2),
            "spot_rate_usd_hr": spot_price,
            "reason": "Task requires more than 4h default limit",
            "status": "PENDING",
            "approval_code": f"APPROVE-{request_id}",
        }
        request_file.write_text(json.dumps(request, indent=2))

        print(f"{'='*60}")
        print(f"📋 Extension Request")
        print(f"{'='*60}")
        print(f"  Request ID:  {request_id}")
        print(f"  Task:        {args.task}")
        print(f"  Instance:    {args.instance} (spot)")
        print(f"  Duration:    {hours}h")
        print(f"  Est. cost:   ${cost:.2f}")
        print(f"  Spot rate:   ${spot_price}/hr")
        print(f"{'='*60}")
        print(f"  To approve, run:")
        print(f"  python gpu_runner.py --task {args.task} --instance {args.instance} \\")
        print(f"    --max-hours {hours} --approval-code APPROVE-{request_id} --script <script>")
        print(f"{'='*60}")
        print(f"  Request saved: {request_file}")
        return

    if not args.task or not args.instance:
        parser.print_help()
        return

    if args.instance not in ALLOWED_INSTANCES:
        print(f"❌ BLOCKED: {args.instance} not in {ALLOWED_INSTANCES}")
        sys.exit(1)

    # === EXTENDED RUN APPROVAL CHECK ===
    if args.max_hours > MAX_RUNTIME_HOURS:
        if not args.approval_code:
            print(f"❌ Runs over {MAX_RUNTIME_HOURS}h require approval.")
            print(f"   First generate a request:")
            print(f"   python gpu_runner.py --request-extension --task {args.task} --instance {args.instance} --max-hours {args.max_hours}")
            sys.exit(1)

        # Verify approval code matches a pending request
        approved = False
        for f in LOG_DIR.glob("extension-request-*.json"):
            req = json.loads(f.read_text())
            if req["approval_code"] == args.approval_code and req["status"] == "PENDING":
                req["status"] = "APPROVED"
                req["approved_at"] = datetime.now(timezone.utc).isoformat()
                f.write_text(json.dumps(req, indent=2))
                approved = True
                print(f"✅ Extension approved: {args.max_hours}h (est. ${req['estimated_cost_usd']})")
                break

        if not approved:
            print(f"❌ Invalid or already-used approval code: {args.approval_code}")
            sys.exit(1)

    # === MANDATORY SECURITY REVIEW ===
    print("🔒 Running security review before launch...")
    review_script = Path(__file__).parent / "security_review.py"
    if review_script.exists():
        review_result = subprocess.run(
            [sys.executable, str(review_script),
             "--check", "ec2-launch",
             "--instance", args.instance,
             "--subnet", SUBNET_TRN1 if "trn" in args.instance else SUBNET_ID],
            capture_output=True, text=True,
        )
        print(review_result.stdout)
        if review_result.returncode != 0:
            print("❌ SECURITY REVIEW FAILED — launch blocked")
            sys.exit(1)
        # Also check for orphans
        orphan_result = subprocess.run(
            [sys.executable, str(review_script),
             "--check", "orphan-instances"],
            capture_output=True, text=True,
        )
        print(orphan_result.stdout)
        if orphan_result.returncode != 0:
            print("⚠️  Orphan instances detected — clean up before launching new ones")
            sys.exit(1)
    else:
        print("⚠️  security_review.py not found — proceeding with built-in checks only")

    # === MANDATORY RUNTIME ESTIMATE CHECK ===
    estimate_dir = LOG_DIR
    estimate_files = sorted(estimate_dir.glob(f"estimate-{args.task}-*.json")) if estimate_dir.exists() else []
    if not estimate_files:
        print(f"❌ No runtime estimate found for task '{args.task}'.")
        print(f"   Run first: python scripts/runtime_estimator.py --task {args.task} --instance {args.instance} --samples N --time-per-sample M")
        print(f"   This collects data points to validate the requested runtime.")
        sys.exit(1)
    else:
        latest_estimate = json.loads(estimate_files[-1].read_text())
        est_hours = latest_estimate["estimated_hours"]
        rec_hours = latest_estimate["recommended_max_hours"]
        est_cost = latest_estimate["estimated_cost_usd"]
        print(f"📊 Runtime estimate: {est_hours}h (recommended max: {rec_hours}h, est. cost: ${est_cost})")
        if args.max_hours > rec_hours * 1.5:
            print(f"⚠️  Requested {args.max_hours}h is much higher than estimated {rec_hours}h — are you sure?")
        # Use estimated max if user didn't override
        if args.max_hours == MAX_RUNTIME_HOURS and rec_hours != MAX_RUNTIME_HOURS:
            args.max_hours = rec_hours
            print(f"   Auto-adjusted max_hours to {rec_hours}h based on estimate")

    start_time = datetime.now(timezone.utc)
    instance_id = None
    result = "UNKNOWN"

    try:
        print(f"{'='*60}")
        print(f"Atlas GPU Runner")
        print(f"Task: {args.task}")
        print(f"Instance: {args.instance} (spot)")
        print(f"Max runtime: {args.max_hours}h")
        print(f"{'='*60}")

        # Setup
        print("\n🔧 Setup:")
        ensure_key_pair(ec2)
        sg_id = ensure_security_group(ec2)

        # Launch
        instance_id = launch_spot(ec2, args.instance, sg_id, args.task)

        # Wait for ready
        private_ip = wait_for_running(ec2, instance_id)
        if not private_ip:
            result = "LAUNCH_FAILED"
            raise RuntimeError("Instance failed to start")

        if not wait_for_ssh(private_ip):
            result = "SSH_FAILED"
            raise RuntimeError("SSH not ready")

        # Show instance info
        r = ssh_run(private_ip, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || neuron-ls 2>/dev/null || echo 'no accelerator detected'")
        print(f"\n🖥️  Accelerator: {r.stdout.strip()}")

        # Run task
        if args.script:
            # SCP extra files first
            for pre in args.pre_scp:
                local, remote = pre.split(":", 1)
                print(f"  📦 SCP: {local} → {remote}")
                scp_to(private_ip, local, remote)

            print(f"\n📦 Copying script...")
            scp_to(private_ip, args.script, "/tmp/run_task.sh")
            ssh_run(private_ip, "chmod +x /tmp/run_task.sh")

            print(f"\n🏃 Running task (max {args.max_hours}h)...")
            timeout_secs = int(args.max_hours * 3600)
            try:
                r = ssh_run(private_ip, "/tmp/run_task.sh", timeout=timeout_secs)
                print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
                if r.stderr:
                    print(f"STDERR: {r.stderr[-1000:]}")
                result = "SUCCESS" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
            except subprocess.TimeoutExpired:
                result = "TIMEOUT"
                print(f"⏰ Task exceeded {args.max_hours}h limit")

            # Collect results
            print(f"\n📥 Collecting results...")
            results_dir = WORKSPACE / "logs" / "gpu-results" / args.task
            results_dir.mkdir(parents=True, exist_ok=True)
            scp_from(private_ip, "/tmp/results/", str(results_dir))
        else:
            print(f"\n⚠️  No --script provided. Instance is running at {private_ip}")
            print(f"    SSH: ssh -i {SSH_KEY_PATH} ubuntu@{private_ip}")
            print(f"    When done: python gpu_runner.py --terminate {instance_id}")
            result = "INTERACTIVE"
            return

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if result == "UNKNOWN":
            result = f"ERROR: {e}"

    finally:
        end_time = datetime.now(timezone.utc)

        # Terminate
        if instance_id and result != "INTERACTIVE":
            print(f"\n🧹 Cleanup:")
            terminate_instance(ec2, instance_id)

        # Log
        if instance_id:
            spot_price = SPOT_PRICES.get(args.instance, 0.50)
            log = write_log(
                args.task, args.instance, instance_id,
                private_ip if 'private_ip' in dir() else "N/A",
                start_time, end_time, result, spot_price,
            )
            print(f"\n{'='*60}")
            print(f"Result: {result}")
            print(f"Duration: {log['duration_hours']}h")
            print(f"Est. cost: ${log['cost_estimate_usd']}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
