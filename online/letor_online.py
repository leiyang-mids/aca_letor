
# from sklearn.feature_extraction.text import *
from sklearn.preprocessing import normalize
import numpy as np

class letor_online:
    ''' runtime component for letor, give plan ranking for query
    '''
    characterizer = None
    centroids = None
    similarity_limit = 0.8

    def __init__(self, query_characterizer, cluster_centroids, similarity = 0.8):
        '''
        query_characterizer : a CountVectorizer object fitted with learned queries
        cluster_centroids   : center points of query clusters
        '''
        # initialize encoder
        self.characterizer = query_characterizer
        self.centroids = cluster_centroids
        self.similarity_limit = similarity

    def get_rank_weight(self, query):
        '''
        get a vector to dot product with indexed ranking in ES from training
        '''
        if type(query) is not str:
            print 'Input parameter query should be a string.'
            return None

        # encode the query, if it's orthogonal to all learnt queries, no ranking
        vQuery = self.characterizer.transform([query])
        if vQuery.data.size == 0:
            print 'No similar query is found, no ranking from LETOR'
            return None

        # set all values to 1 of encoded query (don't care duplicate terms in query)
        vQuery.data = np.ones(vQuery.data.size)

        # find the most similar query, by cosine distance
        similarity = normalize(vQuery.astype(np.float)).dot(self.centroids.T)
        candidates = similarity > self.similarity_limit
        # if no one above limit, return the synthesized ranking from the closest queries
        if candidates.sum() == 0:
            candidates = similarity == similarity.max()

        # weight vector for dot product
        sum_sim = similarity[candidates].sum()
        return [s/sum_sim if c else 0 for s,c in zip(similarity.toarray()[0],candidates.toarray()[0])]
