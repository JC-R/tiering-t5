"""
This is runnable for Milvus(0.11.x) and pymilvus(0.3.x).
"""
import random
import csv
from pprint import pprint

from milvus import Milvus, DataType


class NYUMilvus():

    def __init__(self, name, host, port, collection_param):
        self.host = host
        self.port = port
        self.client = Milvus(host, port)
        self.collection_name = name
        self.collection_param = collection_param

        # self.collection_param = {
        #     "fields": [
        #         {"name": "release_year", "type": DataType.INT32},
        #         {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 8}},
        #     ],
        #     "segment_row_limit": 4096,
        #     "auto_id": False
        # }

    def create_collection(self):
        if self.collection_name not in self.client.list_collections():
            # self.client.drop_collection(self.collection_name)
            self.client.create_collection(self.collection_name, self.collection_param)

        # ------
        # Basic create index:
        #     Now that we have a collection in Milvus with `segment_row_limit` 4096, we can create index or
        #     insert entities.
        #
        #     We can call `create_index` BEFORE we insert any entities or AFTER. However Milvus won't actually
        #     start build index task if the segment row count is smaller than `segment_row_limit`. So if we want
        #     to make Milvus build index, we need to insert number of entities larger than `segment_row_limit`.
        #
        #     We are going to use data in `films.csv` so you can checkout the structure. And we need to group
        #     data with same fields together, so here is a example of how we obtain the data in files and transfer
        #     them into what we need.
        # ------

        ids = []  # ids
        titles = []  # titles
        release_years = []  # release year
        embeddings = []  # embeddings
        films = []
        with open('films.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            films = [film for film in reader]
        for film in films:
            ids.append(int(film[0]))
            titles.append(film[1])
            release_years.append(int(film[2]))
            embeddings.append(list(map(float, film[3][1:][:-1].split(','))))

        hybrid_entities = [
            {"name": "release_year", "values": release_years, "type": DataType.INT32},
            {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
        ]

        # ------
        # Basic insert:
        #     After preparing the data, we are going to insert them into our collection.
        #     The number of films inserted should be 8657.
        # ------
        ids = self.client.insert(self.collection_name, hybrid_entities, ids)
        self.client.flush([self.collection_name])
        after_flush_counts = self.client.count_entities(self.collection_name)
        print(" > There are {} films in collection `{}` after flush".format(after_flush_counts, self.collection_name))

        # ------
        # Basic create index:
        #     Now that we have inserted all the films into Milvus, we are going to build index with these data.
        #
        #     While building index, we have to indicate which `field` to build index for, the `index_type`,
        #     `metric_type` and params for the specific index type. In our case, we want to build a `IVF_FLAT`
        #     index, so the specific params are "nlist". See pymilvus documentation
        #     (https://milvus-io.github.io/milvus-sdk-python/pythondoc/v0.3.0/index.html) for `index_type` we
        #     support and the params accordingly.
        #
        #     If there are already index for a collection and you call `create_index` with different params, the
        #     older index will be replaced by new one.
        # ------
        self.client.create_index(self.collection_name, "embedding",
                                 {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})

        # ------
        # Basic create index:
        #     We can get the detail of the index  by `get_collection_info`.
        # ------
        info = self.client.get_collection_info(self.collection_name)
        pprint(info)

        # ------
        # Basic hybrid search entities:
        #     If we want to use index, the specific index params need to be provided, in our case, the "params"
        #     should be "nprobe", if no "params" given, Milvus will complain about it and raise a exception.
        # ------
        # query_embedding = [random.random() for _ in range(8)]
        # query_hybrid = {
        #     "bool": {
        #         "must": [
        #             {
        #                 "term": {"release_year": [2002, 1995]}
        #             },
        #             {
        #                 "vector": {
        #                     "embedding": {"topk": 3,
        #                                   "query": [query_embedding],
        #                                   "metric_type": "L2",
        #                                   "params": {"nprobe": 8}}
        #                 }
        #             }
        #         ]
        #     }
        # }

        # ------
        # Basic hybrid search entities
        # ------
        # results = client.search(collection_name, query_hybrid, fields=["release_year", "embedding"])
        # for entities in results:
        #     for topk_film in entities:
        #         current_entity = topk_film.entity
        #         print("==")
        #         print("- id: {}".format(topk_film.id))
        #         print("- title: {}".format(titles[topk_film.id]))
        #         print("- distance: {}".format(topk_film.distance))
        #
        #         print("- release_year: {}".format(current_entity.release_year))
        #         print("- embedding: {}".format(current_entity.embedding))

        # ------
        # Basic delete index:
        #     You can drop index for a field.
        # ------
        self.client.drop_index(self.collection_name, "embedding")

        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)

        # ------
        # Summary:
        #     Now we've went through some basic build index operations, hope it's helpful!
        # ------
