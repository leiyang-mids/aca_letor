from s3_helpers import *
from get_rank_for_state_plan import *
from query_characterizer import *
import pickle


def train_one_state(click_data, state):
    '''
    '''
    s3clnt = s3_helper()
    print 'characterize queries for state %s' %state
    s_rows = click_data[click_data['state']==state]    
    q_cluster, q_characterizer, centroids = query_characterizer(s_rows['query'])
    print 'run letor training for state %s' %state
    letor_rank, plans = get_rank_for_state_plan(q_cluster, np.array([[r['ranks'],r['clicks']] for r in s_rows]))
    if not plans or not letor_rank:
        print 'no feature file found for state %s, skip training.' %state
        return
    print 'save ranking & online file on s3'
    save_training = 'training/%s_%d.pickle' %(state, len(letor_rank))
    with open(save_training, 'w') as f:
        pickle.dump([plans, letor_rank], f)
    s3clnt.delete_by_state('training/%s' %state)
    s3clnt.upload(save_training)
    save_online = 'online/%s_runtime.pickle' %state
    with open(save_online, 'w') as f:
        pickle.dump([q_characterizer, centroids], f)
    s3clnt.delete_by_state('online/%s' %state)
    s3clnt.upload(save_online)
