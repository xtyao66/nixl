# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import os
import time

import torch
import torch.distributed as dist

try:
    import nixl_ep_cu13 as nixl_ep
except ModuleNotFoundError:
    import nixl_ep
from test_ht import TCP_STORE_PORT, run_server, test_main
import store_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torchrun high-throughput EP test for GB200 workers with fewer than 8 local GPUs."
    )
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk-groups", type=int, default=None)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help=(
            "Total experts. Defaults to the smallest world-size-compatible "
            "multiple at least 256 so odd GB200 group counts, such as 3x4 "
            "and 5x4, exercise HT instead of failing the base test shape assert."
        ),
    )
    parser.add_argument("--test-ll-compatibility", action="store_true")
    parser.add_argument(
        "--tcp-server",
        type=str,
        default=None,
        help="TCPStore server for NIXL metadata. Defaults to MASTER_ADDR and is started by global rank 0.",
    )
    parser.add_argument(
        "--nvl-group-size",
        type=int,
        default=4,
        help="Ranks per CUDA-IPC/NVLink-local EP group. Use 4 for 4-GPU GB200 workers.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one BF16/no-top-k HT dispatch+combine roundtrip, then exit.",
    )
    parser.add_argument(
        "--debug-smoke-summary",
        action="store_true",
        help="Print recv_x/prefix summaries before smoke data assertions fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert 0 < args.nvl_group_size <= 8 and 8 % args.nvl_group_size == 0

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", args.nvl_group_size))
    rank = int(os.environ["RANK"])
    num_ranks = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))

    init_kwargs = {"backend": "nccl", "init_method": "env://"}
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**init_kwargs)
    group = dist.new_group(list(range(num_ranks)))

    if rank == 0 and args.tcp_server is None:
        server_process = torch.multiprocessing.Process(target=run_server, daemon=True)
        server_process.start()
        time.sleep(0.5)

    dist.barrier()
    tcp_store = store_group.create_client_store(
        master_addr=args.tcp_server or master_addr,
        port=TCP_STORE_PORT,
    )

    num_rdma_groups = max(1, num_ranks // args.nvl_group_size)
    if args.num_topk_groups is None:
        args.num_topk_groups = min(num_rdma_groups, 4)
    if args.num_experts is None:
        args.num_experts = ((256 + num_ranks - 1) // num_ranks) * num_ranks

    num_sms = 24
    ll_num_experts = 256 if args.test_ll_compatibility else 0
    num_qps_per_rank = max(num_sms // 2, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    buffer = nixl_ep.Buffer(
        rank=rank,
        low_latency_mode=False,
        explicitly_destroy=True,
        group=group,
        tcp_store_group=tcp_store,
        nvl_group_size=args.nvl_group_size,
    )
    buffer.update_memory_buffers(
        num_ranks=num_ranks,
        num_experts_per_rank=num_qps_per_rank,
        num_nvl_bytes=int(2e9),
        num_rdma_bytes=int(1e9),
    )
    buffer.connect_ranks([i for i in range(num_ranks) if i != rank])

    if rank == 0 and local_world_size != args.nvl_group_size:
        print(
            f"[warning] LOCAL_WORLD_SIZE={local_world_size} differs from nvl_group_size={args.nvl_group_size}; "
            "rank ordering must still keep each NVL group CUDA-IPC local",
            flush=True,
        )

    torch.manual_seed(rank)
    test_main(
        args,
        num_sms,
        local_rank,
        local_world_size,
        num_ranks,
        num_rdma_groups,
        rank,
        buffer,
        group,
    )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
