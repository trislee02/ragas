from __future__ import annotations

import os
import typing as t
import logging
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import trace_as_chain_group
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager


QUESTION_GEN = HumanMessagePromptTemplate.from_template(
    """
Generate question for the given answer.
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?

Answer:{answer}
Question:
"""  # noqa: E501
)


@dataclass
class AnswerRelevancy(MetricWithLLM):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qa
    batch_size: int = 15
    strictness: int = 3
    embeddings: Embeddings | None = None

    def __post_init__(self: t.Self):
        if self.embeddings is None:
            oai_key = os.getenv("OPENAI_API_KEY", "no-key")
            self.embeddings = OpenAIEmbeddings(openai_api_key=oai_key)  # type: ignore

    def init_model(self):
        super().init_model()

        if isinstance(self.embeddings, OpenAIEmbeddings):
            if self.embeddings.openai_api_key == "no-key":
                raise OpenAIKeyNotFound

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        questions, answers = dataset["question"], dataset["answer"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for ans in answers:
                human_prompt = QUESTION_GEN.format(answer=ans)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            results = self.llm.generate(
                prompts,
                n=self.strictness,
                callbacks=batch_group,
            )
            results = [[i.text for i in r] for r in results.generations]

            scores = []
            for question, gen_questions in zip(questions, results):
                gen_questions_str = "\n".join(gen_questions)
                logging.info(f"ANSWER RELEVANCE:\nquestion:\n{question}\n\ngen_questions:\n{gen_questions_str}")
                cosine_sim = self.calculate_similarity(question, gen_questions)
                scores.append(cosine_sim.mean())

        return scores

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        )
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )


answer_relevancy = AnswerRelevancy()
