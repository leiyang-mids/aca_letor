from s3_helpers import *
from get_rank_for_state_plan import *
from query_characterizer import *
import pickle


def train_one_state(click_data, state, log):
    '''
    '''
    # set folder name of S3
    s3_train, s3_online, s3_fea, s3clnt = 'training', 'online', 'feature_1', s3_helper()
    log.trace('characterize queries for state %s' %state)
    s_rows = click_data[click_data['state']==state]
    q_cluster, vocab, centroids = query_characterizer(s_rows['query'], log)
    log.trace('run letor training for state %s' %state)
    letor_rank, plans = get_rank_for_state_plan(q_cluster, [[r['ranks'],r['clicks']] for r in s_rows], log, s3_fea)
    if not plans: # or (not letor_rank):
        log.warning('no feature file found for state %s, skip training.' %state)
        return

    save_training = '%s/%s_%d.pickle' %(s3_train, state, len(letor_rank))
    with open(save_training, 'w') as f:
        pickle.dump([plans, letor_rank], f)
    s3clnt.delete_by_state('%s/%s' %(s3_train, state))
    s3clnt.upload(save_training)
    save_online = '%s/%s_runtime.pickle' %(s3_online, state)
    with open(save_online, 'w') as f:
        pickle.dump([vocab, centroids], f)
    s3clnt.delete_by_state('%s/%s' %(s3_online, state))
    s3clnt.upload(save_online)
    log.trace('ranking & online file are saved on s3')
