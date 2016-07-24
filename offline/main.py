from datetime import datetime, timedelta
from s3_helpers import *
from get_rank_for_state_plan import *
from get_click_data import *
from query_characterizer import *
import traceback, pickle, time
import numpy as np

from simulate_clicks import *

def main():
	'''
	'''
	next_run, hour, minute = datetime.now(), 5, 10
	s3clnt, log = s3_helper(), logger('training')

	while True:

		if datetime.now() < next_run:
			time.sleep((next_run-datetime.now()).seconds)
			continue
		# get click-through data
		try:
			print 'get query and click data'
			click_data = get_click_data()
		except Exception as ex:
			traceback.print_exc(file=log.log_handler())
			print 'error getting click data, retry in %d minutes.' %minute
			next_run = datetime.now() + timedelta(minutes=minute)
			continue
		# train for each state
		for state in np.unique(click_data['state']):
			try:
				s_rows = click_data[click_data['state']==state]
				print 'characterize queries for the state'
				q_cluster, q_characterizer, centroids = query_characterizer(s_rows['query'])
				print 'run letor training for the state'
				letor_rank, plans = get_rank_for_state_plan(q_cluster, np.array([[r['ranks'],r['clicks']] for r in s_rows]))
				if not plans or not letor_rank:
					print 'no feature file found for state %s, skip training.' %state
					continue
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
			except KeyboardInterrupt:
				sys.exit('User termination')
			except Exception as ex:
				traceback.print_exc(file=log.log_handler())
				print 'training has encountered an error for state %s' %state
		# training completed, get next run time
		next_run = datetime.now() + timedelta(hours=hour)
		print 'training has completed, next run time is %s' %str(next_run)
		# upload log file to S3
		s3clnt.upload2(log.log_name(), 'log/'+log.log_name())

if __name__ == "__main__":
	main()
