# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/NVIDIA-NeMo/Gym
# (resources_servers/workplace_assistant), licensed under Apache-2.0.
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
"""Vendored WorkBench toolkits (NVIDIA nemo-gym ``workplace_assistant``).

Each tool holds an in-memory ``pandas`` snapshot of a workplace database
(email / calendar / analytics / project-management / CRM / company directory)
that is seeded from the CSVs under ``verl/utils/workbench/csv_data`` and mutated
by the tool functions. Comparing the final database state after replaying a
tool-call trajectory against the ground-truth trajectory is what drives the
WorkBench reward (see ``verl.utils.workbench.verify``).
"""
