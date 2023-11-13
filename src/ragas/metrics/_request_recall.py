from __future__ import annotations

import typing as t
import logging
from dataclasses import dataclass

from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

REQUEST_RECALL_RA = HumanMessagePromptTemplate.from_template(
    """
Given a message, and a list of requests, analyze each request and classify if the request is mentioned in the given message or not.
Think in steps and reason before coming to conclusion. 

message: We are experiencing an issue with the head pitch of the QTRobot. Occassionally, the robot tilts its head forward by approximately 10 degrees when the arms move during our experiment. The event seems to occur randomly, and we haven't been able to recreate it. We haven't taken a video yet, but we will send it to you when we do. Please advice me on how to resolve this issue.
requests: 
The robot occasionally tilts its head forward by approximately 10 degrees when the arms move. How to resolve this issue?
The robot occasionally tilts its head forward by approximately 10 degrees when the arms move. What steps can we take to recreate the issue?
Can you provide me a video that shows how the robot occasionally tilts its head forward by approximately 10 degrees when the arms move? 
classification:
1. The robot occasionally tilts its head forward by approximately 10 degrees when the arms move. How to resolve this issue? The request is mentioned clearly in the last sentence of the message. So [Mentioned]
2. The robot occasionally tilts its head forward by approximately 10 degrees when the arms move. What steps can we take to recreate the issue? In the message, the writers haven't been able to recreate the issue, and they do not inquire how to do that. So [Not Mentioned]
3. Can you provide me a video that shows how the robot occasionally tilts its head forward by approximately 10 degrees when the arms move? The message does not ask for a video of the issue. So [Not Mentioned]

message:{message}
requests:
{requests}
classification:
"""  # noqa: E501
)


@dataclass
class RequestRecall(MetricWithLLM):

    """
    Estimates request recall using question and requests.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "request_recall"
    evaluation_mode: EvaluationMode = EvaluationMode.gc
    batch_size: int = 15
    log_name: str = "requests"

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        verdict_token = "[Mentioned]"
        prompts = []
        request, message = dataset["request"], dataset["question"]

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for rq, msg in zip(request, message):
                rq = "\n".join(rq) if isinstance(rq, list) else rq
                human_prompt = REQUEST_RECALL_RA.format(requests=rq, message=msg)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]
            scores = []
            logs = []
            for response in responses:
                logging.info("\n\n\n")
                logging.info(f"REQUEST RECALL: classification:\n{response[0]}")
                sentences = response[0].split("\n")
                denom = len(sentences)
                numerator = sum(
                    bool(sentence.find(verdict_token) != -1) for sentence in sentences
                )
                scores.append(numerator / denom)
                logs.append(response[0])

        return scores, logs


request_recall = RequestRecall()
