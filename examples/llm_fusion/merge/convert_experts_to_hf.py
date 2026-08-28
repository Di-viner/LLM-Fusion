# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP -> HuggingFace conversion that PRESERVES fp32.

verl's built-in ``python -m verl.model_merger merge`` reassembles FSDP shards
but casts every tensor to bfloat16. The training shards are stored in fp32
(mixed-precision fp32 master weights), and the RL task vectors (theta_expert -
theta_base) are small enough that a fp32->bf16 round-trip can add quantization
noise comparable to the delta itself, which changes what a task-vector merge
produces. Every Merge result in the paper was computed from fp32 experts.

This script subclasses verl's FSDPModelMerger and overrides ONLY the two places
that force bfloat16, so the exported HF model stays fp32. Everything else
(world-size detection, DTensor/placement reassembly, tokenizer saving) is reused
unchanged. It does not modify the verl library.

Usage:
    python examples/llm_fusion/merge/convert_experts_to_hf.py \
        --local_dir  <ckpt>/global_step_<N>/actor \
        --target_dir <out>/math
"""

import argparse
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.transformers_compat import drop_tied_target_keys

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


class FP32FSDPModelMerger(FSDPModelMerger):
    """Same as FSDPModelMerger but keeps fp32 instead of casting to bf16."""

    def _load_and_merge_state_dicts(self, world_size, total_shards, mesh_shape, mesh_dim_names):
        from concurrent.futures import ThreadPoolExecutor

        from torch.distributed._tensor import Shard
        from tqdm import tqdm

        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank, lst):
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            lst[rank] = torch.load(model_path, map_location="cpu", weights_only=False)

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards (fp32)", total=total_shards):
                future.result()

        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    # keep fp32 (the ONLY change vs. the upstream bf16 cast)
                    state_dict[key].append(tensor._local_tensor.float())

                    placements = tuple(tensor.placements)
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.float())

        del model_state_dict_lst

        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                continue
            if key in param_placements:
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    assert len(placements) == 1
                    state_dict[key] = self._merge_by_placement(state_dict[key], placements[0])
                else:
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = torch.cat(state_dict[key], dim=0)

        return state_dict

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.float32, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)
        # keep config metadata consistent with the fp32 weights
        model.config.torch_dtype = "float32"

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        drop_tied_target_keys(state_dict, model, self.model_config)

        print(f"Saving fp32 model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        if processor is not None:
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(self.config.target_dir)


def main():
    parser = argparse.ArgumentParser("Convert an FSDP actor checkpoint to a fp32 HF model")
    parser.add_argument("--local_dir", required=True, help="Path to the .../global_step_*/actor dir")
    parser.add_argument("--target_dir", required=True, help="Output HF model dir")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=args.target_dir,
        local_dir=args.local_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
        trust_remote_code=args.trust_remote_code,
    )
    os.makedirs(config.target_dir, exist_ok=True)

    merger = FP32FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":
    main()
