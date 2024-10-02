from __future__ import annotations

from bs4 import BeautifulSoup
from tqdm import tqdm
from html2text import html2text
from dataclasses import dataclass

from src.util import load_json, dump_json

CAS_KOMPETENZEN_PATH = 'CAS_Export_Kompetenzen.json'
CAS_KOMPETENZEN_CLEAN_PATH = 'CAS_Export_Kompetenzen.clean.json'


def process_ja(html: str) -> str:
    # find all tables in the html, if the table is less than 50% of the content, delete it
    soup = BeautifulSoup(html, 'html.parser')
    soup_len = len(str(soup))

    for table in soup.find_all('table'):
        try:
            if len(str(table)) < 0.5 * soup_len:
                table.decompose()
        except:  # noqa
            pass

    return html2text(str(soup))


@dataclass
class CASDocument:
    link: str
    title: str
    abstract: str
    keywords: list[str]


@dataclass
class CASReference:
    branches: list[str]
    descriptions: list[str]
    keywords: list[str]


@dataclass
class CASSample:
    id: str
    name: str
    documents: list[CASDocument]
    jahresabschlüsse: list[str]
    reference: CASReference

    @staticmethod
    def from_json(data: dict) -> CASSample:
        return CASSample(
            id=data['id'],
            name=data['name'],
            documents=[
                CASDocument(
                    link=doc['link'],
                    title=doc['title'],
                    abstract=doc['abstract'],
                    keywords=doc['keywords'],
                )
                for doc in data['documents']
            ],
            jahresabschlüsse=data['jahresabschlüsse'],
            reference=CASReference(
                branches=data['reference']['branches'],
                descriptions=data['reference']['descriptions'],
                keywords=data['reference']['keywords'],
            ),
        )


if __name__ == '__main__':
    dataset = [
        CASSample(
            id=sample['EBID'],
            name=sample['companyname'],
            documents=[
                CASDocument(
                    link=doc['link'],
                    title=doc['title'],
                    abstract=doc['abstract'],
                    keywords=[doc[f'keyword{i}'] for i in range(1, 21) if doc[f'keyword{i}'] != 'null'],
                )
                for doc in sample['payload'][0]['documents']
            ],
            jahresabschlüsse=[
                # process_ja(ja['JA'])
                # for ja in sample['payload'][0]['reference'][0].get('Jahresabschluesse', [])
                # [
                #    # take a random amount of jahresabschlüsse normalverteilt around 5
                #    -round(abs(np.random.normal(5, 2))) :
                # ]
                # if ja['JA_Year'] >= 2018
            ],
            reference=CASReference(
                branches=sample['payload'][0]['reference'][0]['branches'],
                descriptions=[
                    desc['description'] for desc in sample['payload'][0]['reference'][0].get('description', [])
                ],
                keywords=[keyword['keywords'] for keyword in sample['payload'][0]['reference'][0].get('keywords', [])],
            ),
        )
        for sample in tqdm(load_json(CAS_KOMPETENZEN_PATH))
        # if len(sample['payload'][0]['documents']) >= 5 and 'ODS GmbH' not in sample['companyname']
    ]

    dump_json(dataset, CAS_KOMPETENZEN_CLEAN_PATH)
