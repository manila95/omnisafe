# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""Implementation of BC-CPO: CPO with an exponentially decayed constant cost bias."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.second_order.cpo import CPO


@registry.register
class BCCPO(CPO):
    """CPO with a constant bias added to the episode cost.

    The bias shifts the quantity CPO picks its optimization case on, so early updates behave as
    if the policy were more costly than measured. It decays as
    ``cost_bias * cost_bias_decay_rate ** epoch``, so the bias vanishes and BC-CPO converges to
    plain CPO.

    References:
        - Title: Constrained Policy Optimization
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: `CPO <https://arxiv.org/abs/1705.10528>`_
    """

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/EpCostBias')

    def _ep_costs(self) -> float:
        bias = self._cfgs.algo_cfgs.cost_bias * (
            self._cfgs.algo_cfgs.cost_bias_decay_rate ** self._current_epoch
        )
        self._logger.store({'Misc/EpCostBias': bias})
        return super()._ep_costs() + bias
