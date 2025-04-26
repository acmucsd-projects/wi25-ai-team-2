import numpy as np
from flashrank import Ranker, RerankRequest
import json

ranker = Ranker(max_length=128)


def rerank(input): 
	rerankrequest = RerankRequest(query=input, passages=dataset)
	results = ranker.rerank(rerankrequest)

	# https://github.com/PrithivirajDamodaran/FlashRank
	return results