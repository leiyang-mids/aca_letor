# ACA LETOR Runtime Executer

###class
    letor_online(query_characterizer, cluster_centroids)
- query_characterizer : a CountVectorizer object fitted with learned queries
- cluster_centroids   : center points of query clusters
- return              : weights to synthesize ranking from previously seen queries

###runtime data
- located at s3: https://s3.amazonaws.com/w210.data/online/SS_runtime.pickle
- data is saved separately for each state, replace _SS_ in above link with state abbreviation, e.g. OR, NJ etc.
- a simulated file is called _sim_runtime.pickle_
- each file contains **cluster_centroids** and **query_characterizer** for **letor_online** class initialization:

###example
    savedData = 'sim_runtime.pickle'
    with open(savedData) as f:
        qc, c = pickle.load(f)

    letor = letor_online(qc, c)
    rank = letor.get_rank_weight('glaucoma')
