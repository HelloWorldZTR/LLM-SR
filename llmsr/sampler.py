# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib
import requests
import json
import http.client
import os
from openai import OpenAI



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            config: config_lib.Config,
            llm_class: Type[LLM] = LLM,
    ):
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(config, config.api_model)
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self.config.max_sample_num and self.__class__._global_samples_nums >= self.config.max_sample_num:
                break
            
            prompt = self._database.get_prompt()
            print("="*20)
            print("Prompt generated")
            print("-"*20)
            print(prompt.code)
            print("-"*20)
            print("Prompt generated")
            print("="*20)
            
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self.config.samples_per_prompt

            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )
                if cur_global_sample_nums % self.config.snapshot_interval == 0:
                    chosen_evaluator.test_best(**kwargs)

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, config: config_lib.Config, model: str, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(config.samples_per_prompt)
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._model = model
        self._samples_per_prompt = config.samples_per_prompt
        self._instruction_prompt = instruction_prompt
        self._trim = trim
        self._openai_client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.api_base_url,
        )


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])
        
        for _ in range(self._samples_per_prompt):
            try:
                res = self._openai_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._instruction_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=config.max_tokens,
                )

                response = res.choices[0].message.content
                if self._trim:
                    response = _extract_body(response, config)
                all_samples.append(response)
            except Exception as e:
                print(f"Error during API request: {e}")
                continue

        return all_samples