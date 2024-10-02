import os
import json
import requests

from src.util import write_to_file


class JsonBin:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get(self, route: str) -> dict | list[dict]:
        print('Getting jsonbin data from', 'https://api.jsonbin.io/v3/' + route)
        response = requests.get(
            f'https://api.jsonbin.io/v3/{route}',
            headers={
                'X-Master-Key': self.api_key,
                'X-Sort-Order': 'ascending',
                'Content-Type': 'application/json',
            },
            json={},
        )

        return json.loads(response.text)

    def bins(self) -> list[str]:
        # loads all the uncategorized bin ids from the jsonbin api and returns them as a list
        bins: list[str] = []

        if os.path.exists('jsonbin_cache/bins.json'):
            with open('jsonbin_cache/bins.json', 'r') as f:
                bins: list[str] = json.loads(f.read())
        else:
            ten_bins = self.get('c/uncategorized/bins')
            for bin in ten_bins:
                bins.append(bin['record'])

        while True:
            ten_bins = self.get(f'c/uncategorized/bins/{bins[-1]}')
            for bin in ten_bins:
                bins.append(bin['record'])
            if len(ten_bins) < 10:
                break

        write_to_file('jsonbin_cache/bins.json', json.dumps(bins))

        return bins

    def bin(self, bin_id: str) -> dict:
        if os.path.exists(f'jsonbin_cache/{bin_id}.json'):
            with open(f'jsonbin_cache/{bin_id}.json', 'r') as f:
                return json.loads(f.read())

        data = self.get(f'b/{bin_id}')['record']  # type: ignore

        write_to_file(f'jsonbin_cache/{bin_id}.json', json.dumps(data))

        return data

    def get_all_feedback(self) -> list[str]:
        feedback: list[str] = []

        for bin_id in self.bins():
            data = self.bin(bin_id)
            for item in data:
                feedback.append(item['feedback'] or '')

        return feedback
