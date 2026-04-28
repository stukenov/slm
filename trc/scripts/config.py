"""
SozKZ TRC Grant — Configuration
Google TPU Research Cloud project settings and zone allocations.
"""

import os

# Project settings
PROJECT_ID = "sozkz-trc"
PROJECT_NUMBER = "65934168684"
SERVICE_ACCOUNT = "sozkz-tpu@sozkz-trc.iam.gserviceaccount.com"

# Credentials
CREDENTIALS_PATH = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/slm/trc/credentials/sozkz-tpu-key.json")
)

# TPU quota allocations — ONLY create TPUs in these zones!
# Creating in other zones will incur REAL CHARGES.
ALLOCATIONS = [
    {
        "name": "v6e-us-east1",
        "tpu_type": "v6e",
        "accelerator_type": "v6e-64",
        "chips": 64,
        "quota_type": "spot",
        "zone": "us-east1-d",
    },
    {
        "name": "v4-spot-us-central2",
        "tpu_type": "v4",
        "accelerator_type": "v4-32",
        "chips": 32,
        "quota_type": "spot",
        "zone": "us-central2-b",
    },
    {
        "name": "v4-ondemand-us-central2",
        "tpu_type": "v4",
        "accelerator_type": "v4-32",
        "chips": 32,
        "quota_type": "on-demand",
        "zone": "us-central2-b",
    },
    {
        "name": "v5e-us-central1",
        "tpu_type": "v5e",
        "accelerator_type": "v5litepod-64",
        "chips": 64,
        "quota_type": "spot",
        "zone": "us-central1-a",
    },
    {
        "name": "v6e-europe-west4",
        "tpu_type": "v6e",
        "accelerator_type": "v6e-64",
        "chips": 64,
        "quota_type": "spot",
        "zone": "europe-west4-a",
    },
    {
        "name": "v5e-europe-west4",
        "tpu_type": "v5e",
        "accelerator_type": "v5litepod-64",
        "chips": 64,
        "quota_type": "spot",
        "zone": "europe-west4-b",
    },
]

# Smallest test TPU configs (for testing connectivity without using full quota)
TEST_ALLOCATIONS = [
    {
        "name": "test-v4-ondemand",
        "accelerator_type": "v4-8",
        "zone": "us-central2-b",
        "spot": False,
        "runtime_version": "tpu-vm-tf-2.17.0-pjrt",
    },
    {
        "name": "test-v4-spot",
        "accelerator_type": "v4-8",
        "zone": "us-central2-b",
        "spot": True,
        "runtime_version": "tpu-vm-tf-2.17.0-pjrt",
    },
    {
        "name": "test-v5e-us",
        "accelerator_type": "v5litepod-4",
        "zone": "us-central1-a",
        "spot": True,
        "runtime_version": "v2-alpha-tpuv5-lite",
    },
    {
        "name": "test-v5e-eu",
        "accelerator_type": "v5litepod-4",
        "zone": "europe-west4-b",
        "spot": True,
        "runtime_version": "v2-alpha-tpuv5-lite",
    },
    {
        "name": "test-v6e-us",
        "accelerator_type": "v6e-4",
        "zone": "us-east1-d",
        "spot": True,
        "runtime_version": "v2-alpha-tpuv6e",
    },
    {
        "name": "test-v6e-eu",
        "accelerator_type": "v6e-4",
        "zone": "europe-west4-a",
        "spot": True,
        "runtime_version": "v2-alpha-tpuv6e",
    },
]

# All zones we have quota in
ALL_ZONES = list(set(a["zone"] for a in ALLOCATIONS))

# Custom subnet names per zone (us-central2 has non-default subnet)
SUBNET_OVERRIDES = {
    "us-central2-b": "default-us-central2",
}

def get_subnet_flag(zone):
    """Return --subnetwork flag value if zone needs a custom subnet, else None."""
    return SUBNET_OVERRIDES.get(zone)

# Runtime versions by TPU type
RUNTIME_VERSIONS = {
    "v4": "tpu-vm-tf-2.17.0-pjrt",
    "v5e": "v2-alpha-tpuv5-lite",
    "v6e": "v2-alpha-tpuv6e",
}
