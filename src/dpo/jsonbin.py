import json
import requests


class JsonBin:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get(self, route: str) -> dict | list[dict]:
        print('Getting jsonbin data from', 'https://api.jsonbin.io/v3/' + route)
        response = requests.get(
            f'https://api.jsonbin.io/v3/{route}',
            headers={
                'X-Master-Key': self.api_key,
                'Content-Type': 'application/json',
            },
            json={},
        )

        return json.loads(response.text)

    def bins(self) -> list[str]:
        # loads all the uncategorized bin ids from the jsonbin api and returns them as a list
        bins: list[str] = []

        ten_bins = self.get('c/uncategorized/bins')
        for bin in ten_bins:
            bins.append(bin['record'])

        last_requested = ''

        while last_requested != bins[-1] and len(ten_bins) == 10:
            last_requested = bins[-1]
            ten_bins = self.get(f'c/uncategorized/bins/{last_requested}')
            for bin in ten_bins:
                bins.append(bin['record'])

        return bins

    def bin(self, bin_id: str) -> dict:
        return self.get(f'b/{bin_id}')['record']  # type: ignore
