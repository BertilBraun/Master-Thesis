from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

from docarray import BaseDoc
from docarray.typing import NdArray

from gpt_util import create_embeddings
from src.types import Example


class DBDoc(BaseDoc):
    example: Example
    embedding: NdArray[384]


# Specify your workspace path
db = InMemoryExactNNVectorDB[DBDoc](workspace='./workspace_path')


class DB:
    @staticmethod
    def insert(examples: list[Example] | Example):
        if isinstance(examples, Example):
            examples = [examples]

        embeddings = create_embeddings(list(map(str, examples)))
        docs = [DBDoc(example=example, embedding=embedding) for example, embedding in zip(examples, embeddings)]
        
        db.index(docs=DocList[DBDoc](docs))

    @staticmethod
    def 

# Perform a search query
query = DBDoc(text='query', embedding=np.random.rand(128))
results = db.search(inputs=DocList[DBDoc]([query]), limit=10)

# Print out the matches
for m in results[0].matches:
    print(m)
