# Evaluator script that takes in a data, model port and outputs the MCQ and Free Response metrics for the dataset

import argparse
import json
import os
import pandas as pd
import asyncio

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.metrics import MCQMetrics, FreeResponseMetrics


class Evaluator:
    def __init__(self, data, port_number=8080):
        self.data = data
        self.port_number = port_number

    async def evaluate_row(self, row):
        # Extract relevant information from the dataframe row
        question = row['question']
        reference = row['reference']
        options = row['options']

        if 'response' in row:
            response = row['response']
        else:
            raise Exception('Response not found in the dataframe. Please provide a response column in the dataframe.')

        # Create instances of metrics classes
        mcq_metrics = MCQMetrics(question, reference, options)
        free_response_metrics = FreeResponseMetrics(question=question, reference=reference, response=response)

        # Run the metrics asynchronously
        mcq_result = await mcq_metrics()
        free_response_result = await free_response_metrics()

        return {
            'mcq_metrics': mcq_result,
            'free_response_metrics': free_response_result,
        }

    async def evaluate_dataframe(self):
        results = []

        for index, row in self.data.iterrows():
            result = await self.evaluate_row(row)
            results.append(result)

        return results

    async def run_evaluation(self):
        results = await self.evaluate_dataframe()
        print(results)

if __name__ == "__main__":

    data = {
        'question': ["What is the capital of France?", "Who wrote Hamlet?", "What is Chandler Bing's Middle Name?"],
        'reference': ["Paris", "William Shakespeare", "Muriel"],
        'options': [["A. Paris", "B. Rome", "C. Madrid", "D. Berlin"],
                    ["A. William Shakespeare", "B. Charles Dickens", "C. Jane Austen", "D. Mark Twain"],
                    ["A. Meredith", "B. Muriel", "C. Richard", "D. Robert"]],
        'response': ["Paris", "William Shakespeare", "B"],
    }

    df = pd.DataFrame(data)
    port_number = 8080

    evaluator = Evaluator(data=df, port_number=port_number)
    
    print(asyncio.run(evaluator.run_evaluation()))
        