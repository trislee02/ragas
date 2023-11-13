"""
Q - question
A - answer: generated_text from RAG pipeline
C - contexts: context used for generation
G - ground_truths: ground truth answer
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import floor

from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from tqdm import tqdm

from ragas.exceptions import OpenAIKeyNotFound
from ragas.llms import LangchainLLM, llm_factory

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks


def make_batches(total_size: int, batch_size: int) -> list[range]:
    """
    Take a total size and batch size and return a list of ranges for the batches
    """
    tail = total_size % batch_size
    num_batches = floor(total_size / batch_size)
    batches = [
        range(i, i + batch_size) for i in range(0, batch_size * num_batches, batch_size)
    ]
    if tail != 0:
        batches.append(range(batch_size * num_batches, batch_size * num_batches + tail))

    return batches


EvaluationMode = Enum("EvaluationMode", "qac qa qc gc ga qga")


@dataclass
class Metric(ABC):
    batch_size: int
    verbose: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def log_name(self) -> str:
        ...

    @property
    @abstractmethod
    def evaluation_mode(self) -> EvaluationMode:
        ...

    @abstractmethod
    def init_model():
        """
        This method will lazy initialize the model.
        """
        ...

    def score(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
    ) -> Dataset:
        scores = []
        logs = []
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
        with trace_as_chain_group(f"ragas_{self.name}", callback_manager=cm) as group:
            for batch in tqdm(self.get_batches(len(dataset))):
                score, log = self._score_batch(dataset.select(batch), callbacks=group)
                scores.extend(score)
                if log:
                    logs.extend(log)
        if len(logs) == len(scores):
            dataset_log = dataset.add_column(f"{self.log_name}", logs)
        else:
            print("None dataset log")
            dataset_log = None
        return dataset.add_column(f"{self.name}", scores), dataset_log  # type: ignore

    @abstractmethod
    def _score_batch(
        selfself: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[Callbacks] = None,
        callback_group_name: str = "batch",
    ) -> tuple[list, list]:
        ...


    def score_single(
        self: t.Self,
        ds_row: dict,
        callbacks: t.Optional[Callbacks] = None,
    ) -> float:
        """
        Score for a single row of dataset
        """
        # TODO: validation check if they are string

        ds = Dataset.from_dict({k: [v] for k, v in ds_row.items()})
        score = self._score_batch(
            ds, callback_group_name=self.name, callbacks=callbacks
        )

        return score[0]

    def get_batches(self, dataset_size: int) -> list[range]:
        return make_batches(dataset_size, self.batch_size)


@dataclass
class MetricWithLLM(Metric):
    llm: LangchainLLM = field(default_factory=llm_factory)

    def init_model(self):
        if isinstance(self.llm, ChatOpenAI) or isinstance(self.llm, OpenAI):
            self.llm.langchain_llm = t.cast(ChatOpenAI, self.llm)
            if self.llm.langchain_llm.openai_api_key == "no-key":
                raise OpenAIKeyNotFound
