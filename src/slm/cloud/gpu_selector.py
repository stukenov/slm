"""GPU selection and cost estimation for vast.ai instances."""

from __future__ import annotations

import logging

from . import vastai

log = logging.getLogger(__name__)

# Approximate throughput (iterations/sec) for a ~50M param LLM
# with batch_size=16, block_size=512, single GPU.
GPU_THROUGHPUT: dict[str, float] = {
    "RTX 3060":  2.5,
    "RTX 3090":  4.5,
    "RTX 4090":  7.0,
    "RTX A4000": 3.5,
    "RTX A5000": 4.8,
    "A10":       3.2,
    "A100":      8.0,
    "A100_SXM4": 8.5,
    "A6000":     5.0,
    "L40":       6.5,
    "L40S":      7.0,
    "H100":     12.0,
    "H200":     14.0,
    "RTX 5090": 10.0,
    "RTX 5080":  8.0,
}

# Minimum requirements for bf16 training
MIN_COMPUTE_CAP = 800   # sm_80+ for bf16
MIN_GPU_RAM_GB = 12
MIN_RELIABILITY = 0.90
MIN_INET_DOWN = 500      # Mbps — filter out slow connections
MIN_DISK_SPACE = 15      # GB — minimal; dataset downloaded at runtime

# Blackwell GPUs that are NOT yet supported (5090 works with CUDA 12.8+)
BLACKWELL_GPUS = {"RTX 5070 Ti", "RTX 5070", "RTX 5060 Ti"}


def _estimate_hours(config: dict, throughput: float, num_gpus: int = 1) -> float:
    """Rough estimate of training time in hours.

    With DDP, effective batch = per_device * grad_accum * num_gpus,
    which reduces total steps proportionally.
    """
    dataset_size = 23_600_000  # multidomain-kazakh-dataset train split
    block_size = config.get("block_size", 512)
    batch_size = config.get("per_device_train_batch_size", 16)
    grad_accum = config.get("gradient_accumulation_steps", 2)
    epochs = config.get("num_train_epochs", 1)

    tokens_per_sample = block_size
    total_tokens = dataset_size * 50  # avg ~50 tokens per sample (conservative)
    total_samples = total_tokens // tokens_per_sample
    effective_batch = batch_size * grad_accum * num_gpus
    steps_per_epoch = total_samples // effective_batch
    total_steps = steps_per_epoch * epochs

    # max_steps override
    max_steps = config.get("max_steps", 0)
    if max_steps and max_steps > 0:
        total_steps = min(total_steps, max_steps)

    hours = total_steps / (throughput * 3600)
    return hours


def _gpu_name_matches(offer_gpu: str, known_gpu: str) -> bool:
    """Check if an offer's GPU name matches a known GPU key."""
    offer_lower = offer_gpu.lower().replace("-", " ").replace("_", " ")
    known_lower = known_gpu.lower().replace("-", " ").replace("_", " ")
    return known_lower in offer_lower


def _get_throughput(gpu_name: str) -> float:
    """Look up throughput for a GPU, defaulting to conservative estimate."""
    for known, tput in GPU_THROUGHPUT.items():
        if _gpu_name_matches(gpu_name, known):
            return tput
    return 3.0  # conservative fallback


def select_best_offer(
    config: dict,
    *,
    max_price: float = 0.50,
    num_gpus: int = 1,
    min_disk: int = 0,
    gpu_override: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Find the cheapest offer that meets requirements, ranked by total cost.

    Returns a dict with keys: offer, estimated_hours, estimated_cost, gpu_name.
    """
    # Note: dph_total filter is unreliable in vast.ai API, so we filter in Python
    query_parts = [
        "rentable=true",
        f"num_gpus>={num_gpus}",
        f"compute_cap>={MIN_COMPUTE_CAP}",
    ]
    if gpu_override:
        # Vast.ai query uses underscores in GPU names
        gpu_query = gpu_override.replace(" ", "_")
        query_parts.append(f"gpu_name={gpu_query}")

    query = " ".join(query_parts)
    log.info("Searching offers: %s", query)

    offers = vastai.search_offers(query=query, order="dph_total")
    # vast.ai dph_total is already the total price for all GPUs
    for o in offers:
        o["_real_dph"] = float(o.get("dph_total", 999))
    offers = [o for o in offers if o["_real_dph"] <= max_price]
    if not offers:
        raise RuntimeError(
            f"No GPU offers found matching criteria (max ${max_price}/hr). "
            "Try increasing --max-price or relaxing GPU requirements."
        )

    # Filter by minimum network speed and reliability
    offers = [o for o in offers if float(o.get("inet_down", 0)) >= MIN_INET_DOWN]
    offers = [o for o in offers if float(o.get("reliability", 0)) >= MIN_RELIABILITY]

    # Filter out Blackwell GPUs (unsupported by current Docker image)
    offers = [o for o in offers if o.get("gpu_name", "") not in BLACKWELL_GPUS]
    if not offers:
        raise RuntimeError(
            "No compatible GPU offers found (Blackwell GPUs excluded). "
            "Try increasing --max-price or relaxing GPU requirements."
        )

    scored = []
    for offer in offers:
        gpu_name = offer.get("gpu_name", "unknown")
        dph = offer["_real_dph"]
        throughput = _get_throughput(gpu_name)
        offer_gpus = int(offer.get("num_gpus", num_gpus))
        hours = _estimate_hours(config, throughput, num_gpus=offer_gpus)
        total_cost = dph * hours
        scored.append({
            "offer": offer,
            "gpu_name": gpu_name,
            "dph": dph,
            "throughput_its": throughput,
            "estimated_hours": round(hours, 1),
            "estimated_cost": round(total_cost, 2),
        })

    scored.sort(key=lambda x: x["estimated_cost"])
    best = scored[0]

    log.info(
        "Best offer: %s @ $%.3f/hr — est. %.1fh, $%.2f total",
        best["gpu_name"], best["dph"],
        best["estimated_hours"], best["estimated_cost"],
    )

    if dry_run:
        print("\n=== Dry Run: Top 5 GPU Offers ===")
        for i, s in enumerate(scored[:5]):
            print(
                f"  {i+1}. {s['gpu_name']:15s} "
                f"${s['dph']:.3f}/hr  "
                f"~{s['estimated_hours']:.1f}h  "
                f"~${s['estimated_cost']:.2f} total  "
                f"({s['throughput_its']:.1f} it/s)"
            )
        print()

    return best
