# Evaluator script that takes in a data, model port and outputs the MCQ and Free Response metrics for the dataset

import argparse
import json
import os
import pandas as pd
import asyncio
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.metrics import MCQMetrics, FreeResponseMetrics


class Evaluator:
    def __init__(self, data, port_number=8080):
        self.data = data
        self.api_url = f"http://localhost:{port_number}/completion"

    async def evaluate_row(self, row):
        # Extract relevant information from the dataframe row
        question = row['question']
        reference = row['reference']
        options = row['options']

        if 'response' in row:
            response = row['response']
        else:
            response = None

        # Create instances of metrics classes
        mcq_metrics = MCQMetrics(question=question, reference=reference, options=options)
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

        for index, row in tqdm(self.data.iterrows(), desc="Evaluating rows", total=len(self.data), ncols=80):
            result = await self.evaluate_row(row)
            results.append(result)

        return results

    async def run_evaluation(self):
        results = await self.evaluate_dataframe()
        
        # print(results)
        return results

if __name__ == "__main__":

    #Main function is for example usage. this script is not meant to be run directly and instead called from other scripts
    # If Data and port are specified in argparse use that else use the example data
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type = str, default='None', help='Path to the data file')
    parser.add_argument('--port', '-p', type = int, default=0, help='Port number for the LLM server')
    args = parser.parse_args()

    if args.data_path != 'None':
        df = pd.read_csv(args.data)
    else:
        data = {
            'question': ["What is the capital of France?", "Who wrote Hamlet?", "What is Chandler Bing's Middle Name?"],
            'reference': ["Paris", "William Shakespeare", "Muriel"],
            'options': [["A. Paris", "B. Rome", "C. Madrid", "D. Berlin"],
                        ["A. William Shakespeare", "B. Charles Dickens", "C. Jane Austen", "D. Mark Twain"],
                        ["A. Meredith", "B. Muriel", "C. Richard", "D. Robert"]],
            'response': ["Paris", "William Shakespeare", "B"],
        }

        df = pd.DataFrame(data)
    
    if args.port != 0:
        port_number = args.port
    else:
        port_number = 8080

    evaluator = Evaluator(data=df, port_number=port_number)

    print(asyncio.run(evaluator.run_evaluation()))
    
