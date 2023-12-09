from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
from nltk.translate.chrf_score import sentence_chrf
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re
import asyncio
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.llm import llm_completion


class MCQMetrics:
    """Computes MCQ metric scores given a LLM Reponse and Reference Text.

    The MCQ metric scores are:
    """
    def __init__(self, question, reference, options, samples=5):
        """Initializes the MCQMetrics object.

        Args:
            question (str): The question.
            reference (str): The reference text.
            options (list): The options.
            samples (int): The number of samples.
        
        Returns:
            dict: The MCQ metric scores.
        """
        
        self.question = question
        self.reference = reference
        self.options = options
        self.samples = samples

    async def __call__(self) -> dict:
        """Computes all MCQ metric scores and append the average score.

        Returns:
            dict: The MCQ metric scores.
        """

        outputs = []
        answers = []
        option_a, option_b, option_c, option_d = self.options

        system_instruction = """
            You are a huge fan of the TV Show Friends. You will be given a QUESTION and four OPTIONS. I want you to ANSWER the QUESTION with the following steps.

            Evaluation Steps:
            1. Read the QUESTION carefully.
            2. Choose the correct OPTION from OPTIONS best of your knowledge.
            3. Output the ANSWER which is a single alphabet from A, B, C, D which is the right OPTION for the QUESTION
            4. The Output format for each OPTION is 
                for A: 'ANSWER: A'
                for B: 'ANSWER: B'
                for C: 'ANSWER: C'
                for D: 'ANSWER: D'

            Here are a few Examples for how I expect the answer to be.
            Examples:

            {
                QUESTION: What is the name of Ross and Rachel's daughter,
                OPTIONS: 
                    A. Emma
                    B. Delilah
                    C. Bemma
                    D. Deliluu
                ANSWER: A
            },

            {
                QUESTION: What is Chandler Bing's Middle Name,
                OPTIONS: 
                    A. Meredith
                    B. Muriel
                    C. Richard
                    D. Robert
                ANSWER: B
            }

            Based on the above Evaluation Steps and Examples now ANSWER the QUESTION I give you
        """

        user_query = f"""

            QUESTION: {self.question}

            OPTIONS: 
                A. {option_a}
                B. {option_b}
                C. {option_c}
                D. {option_d}

            OUTPUT only the ANSWER which is either A, B, C, or D. 
            The ANSWER is
        """
        
        other_answers = []
        while not answers:
            for _ in range(self.samples):
                output = await llm_completion(system_instruction, user_query)
                output = output['content']

                outputs.append(output)

                match = re.search(r'\b(?:ANSWER|Answer is)(?:\:|) ([A-D])\b', output, flags=re.IGNORECASE)

                if match:
                    answer_choice = match.group(1)
                    answers.append(answer_choice)

        # Count occurrences
        correct_format_rate = len(answers) / self.samples
        correct_answer_rate = sum(1 for answer in answers if answer == self.reference) / len(answers)

        # get the most occurring sampled answer as the final answer
        counts = Counter(answers)
        answer = max(counts, key=lambda x: (counts[x], answers.index(x)))

        # print(outputs)
        # print(answers)

        mcq_metrics = {
            'correct_format_rate': correct_format_rate,
            'correct_answer_rate': correct_answer_rate,
            'answer': answer,
        }
        
        return mcq_metrics


class FreeResponseMetrics:
    """Computes free response metric scores given a LLM Reponse and Reference Text.

    The free response metric scores are:
    - BLEU
    - ROUGE-1
    - CHRF
    - Jaccard Similarity
    - Average of the above scores
    """

    def __init__(
        self, 
        question=None, 
        reference=None,
        response=None,
    ):
        """Initializes the FreeResponseMetrics object.

        Args:
            response_text (str): The response text.
            reference_text (str): The reference text.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        
        if response is None:
            raise ValueError('Response text cannot be None.')
        
        self.response_text = response
        self.response_text = ' '.join(tokenizer.tokenize(self.response_text))

        if reference is None:
            raise ValueError('Reference text cannot be None.')
        
        self.reference_text = reference
        self.reference_text = ' '.join(tokenizer.tokenize(self.reference_text))
    
    async def __call__(self) -> dict:
        """Computes all free response metric scores and append the average score.

        Returns:
            dict: The free response metric scores.
        """

        bleu_score = await self.compute_bleu()
        rouge_score = await self.compute_rouge()
        chrf_score = await self.compute_chrf()
        jaccard_score = await self.compute_jaccard()

        return {
            'average_score': (bleu_score + rouge_score + chrf_score + jaccard_score) / 4,
            'bleu': bleu_score,
            'rouge': rouge_score,
            'chrf': chrf_score,
            'jaccard': jaccard_score,
        }

    async def compute_bleu(self) -> float:
        """Computes the BLEU score.

        Returns:
            float: The BLEU score.
        """

        chencherry = SmoothingFunction()

        return float(
            sentence_bleu(
                [self.reference_text.split()],
                self.response_text.split(),
                smoothing_function=chencherry.method1,
                weights=(1, 0, 0, 0)
            )
        )
    
    async def compute_rouge(self) -> float:
        """Computes the ROUGE-1 score.

        Returns:
            float: The ROUGE-1 score.
        """

        rouge = evaluate.load('rouge')
        
        results = rouge.compute(
            predictions=[self.response_text],
            references=[self.reference_text],
            use_aggregator=True
        )

        return float(results['rouge1'])
    
    async def compute_chrf(self) -> float:
        """Computes the CHRF score.

        Returns:
            float: The CHRF score.
        """

        return float(
            sentence_chrf(
                self.reference_text,
                self.response_text
            )
        )
    
    async def compute_jaccard(self) -> float:
        """Computes the Jaccard similarity score.

        Returns:
            float: The Jaccard similarity score.
        """

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_jaccard)

    def _compute_jaccard(self) -> float:
        """Computes the Jaccard similarity score.

        Returns:
            float: The Jaccard similarity score.
        """

        response_docs = set(self.response_text.lower().split())
        reference_docs = set(self.reference_text.lower().split())
        intersection = response_docs.intersection(reference_docs)
        union = response_docs.union(reference_docs)
        return float(len(intersection) / len(union))
