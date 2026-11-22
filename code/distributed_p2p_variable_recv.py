from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("The semantic gate requires exactly two ranks")

    peer = 1 - rank
    send_rows = 3 if rank == 0 else 5
    maximum_rows = 8
    width = 4
    send_count = torch.tensor(
        [send_rows],
        device=device,
        dtype=torch.int64,
    )
    recv_count = torch.full(
        (1,),
        -1,
        device=device,
        dtype=torch.int64,
    )
    send_payload = (
        torch.arange(
            send_rows * width,
            device=device,
            dtype=torch.float32,
        ).reshape(send_rows, width)
        + rank * 1_000
    )
    sentinel = -777.0
    recv_payload = torch.full(
        (maximum_rows, width),
        sentinel,
        device=device,
        dtype=torch.float32,
    )

    operations = [
        dist.P2POp(
            dist.isend,
            send_count,
            peer,
        ),
        dist.P2POp(
            dist.irecv,
            recv_count,
            peer,
        ),
        dist.P2POp(
            dist.isend,
            send_payload,
            peer,
        ),
        dist.P2POp(
            dist.irecv,
            recv_payload,
            peer,
        ),
    ]
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()
    torch.cuda.synchronize()

    received_rows = int(recv_count.item())
    expected_rows = 5 if rank == 0 else 3
    expected_payload = (
        torch.arange(
            expected_rows * width,
            device=device,
            dtype=torch.float32,
        ).reshape(expected_rows, width)
        + peer * 1_000
    )
    payload_equal = torch.equal(
        recv_payload[:received_rows],
        expected_payload,
    )
    tail_untouched = torch.equal(
        recv_payload[received_rows:],
        torch.full_like(
            recv_payload[received_rows:],
            sentinel,
        ),
    )
    local_pass = (
        received_rows == expected_rows
        and payload_equal
        and tail_untouched
    )
    passed = torch.tensor(
        int(local_pass),
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if int(passed.item()) != 1:
        raise RuntimeError(
            "Variable receive semantic gate failed: "
            f"rank={rank}, received={received_rows}, "
            f"payload_equal={payload_equal}, "
            f"tail_untouched={tail_untouched}"
        )

    gathered: list[dict[str, Any] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "send_rows": send_rows,
            "received_rows": received_rows,
            "maximum_rows": maximum_rows,
            "payload_equal": payload_equal,
            "tail_untouched": tail_untouched,
        },
    )
    result = None
    if rank == 0:
        result = {
            "suite": "distributed_p2p_variable_recv",
            "world_size": world_size,
            "backend": dist.get_backend(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "records": gathered,
            "correctness": {
                "all_passed": True,
                "larger_receive_buffer_supported": True,
                "tail_preserved": True,
            },
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(output)

    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
