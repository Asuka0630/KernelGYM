"""
Single GPU Worker launcher for KernelGym.
"""
import asyncio
import argparse
import logging
import os
import sys
import redis.asyncio as redis

from kernelgym.config import settings
KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from kernelgym.worker.gpu_worker import GPUWorker

logger = logging.getLogger("kernelgym.single_worker")


def _ensure_torch_cuda_arch_list() -> None:
    """Pin ``TORCH_CUDA_ARCH_LIST`` to match the compile_service worker."""

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    try:
        from kernelgym.worker.compile_offload import detect_cuda_arch_list
    except Exception:  # noqa: BLE001 - compile_offload is optional.
        return
    arch_list = detect_cuda_arch_list()
    if arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list


async def main():
    """Main entry point for single GPU worker."""
    parser = argparse.ArgumentParser(description="Start a single GPU worker")
    parser.add_argument("--worker-id", required=True, help="Worker ID")
    parser.add_argument("--device", required=True, help="GPU device (e.g., cuda:0)")
    parser.add_argument("--persistent", action="store_true", help="Record process info for persistent monitor")
    args = parser.parse_args()

    # Align ARCH_LIST with compile_service before any subprocess is
    # spawned (spawn context inherits os.environ).
    _ensure_torch_cuda_arch_list()

    # Configure logging
    logger = setup_logging(f"worker_{args.worker_id}")
    logger.info(
        f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '<unset>')}"
    )
    
    # Initialize Redis connection
    redis_client = redis.from_url(settings.redis_url)
    await redis_client.ping()
    logger.info(f"Redis connection established for worker {args.worker_id}")
    
    # Create and start worker
    worker = GPUWorker(args.worker_id, args.device, redis_client)
    
    try:
        logger.info(f"Starting single worker {args.worker_id} on device {args.device}")
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
    finally:
        try:
            # In persistent mode, clear process info on clean exit
            if args.persistent:
                await redis_client.delete(f"{KEY_PREFIX}:worker_process:{args.worker_id}")
        except Exception:
            pass
        await worker.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
