from __future__ import annotations

import typing as t
import logging
from dataclasses import dataclass

from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

CONTEXT_RECALL_RA = HumanMessagePromptTemplate.from_template(
    """
Given a context, and an answer, analyze each sentence (ignore social chitchat, meaningless, and email sentences) in the answer and classify if the sentence can be attributed to the given context or not.
Think in steps and reason before coming to conclusion.

context: 
<context>
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
</context>
answer: 
<answer>
Hello everyone. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895 
</answer>
classification:
1. Hello everyone. This sentence is a greeting. So [Ignored]
2. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
3. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
4. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
5. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]

context:
<context>
{context}
</context>
answer:
<answer>
{ground_truth}
</answer>
classification:
"""  # noqa: E501
)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_recall"
    evaluation_mode: EvaluationMode = EvaluationMode.gc
    batch_size: int = 15
    log_name: str = "sentences"

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        verdict_token = "[Attributed]"
        ignored_token = "[Ignored]"
        prompts = []
        ground_truths, contexts = dataset["ground_truths"], dataset["contexts"]

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for gt, ctx in zip(ground_truths, contexts):
                gt = "\n".join(gt) if isinstance(gt, list) else gt
                ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
                human_prompt = CONTEXT_RECALL_RA.format(context=ctx, ground_truth=gt)
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
                logging.info(f"CONTEXT RECALL: sentences: {response[0]}")
                sentences = response[0].split("\n")
                denom = len(sentences)
                numerator = sum(
                    bool(sentence.find(verdict_token) != -1) for sentence in sentences
                )
                ignored = sum(
                    bool(sentence.find(ignored_token) != -1) for sentence in sentences
                )
                scores.append(numerator / (denom - ignored + 0.000001)) 
                logs.append(response[0])
                
        return scores, logs
    
context_recall = ContextRecall()
