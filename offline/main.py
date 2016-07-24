from datetime import datetime, timedelta
from get_click_data import *
from train_one_state import *
import traceback, time
import numpy as np

from simulate_clicks import *

def main():
	'''
	'''
	next_run, hour, minute = datetime.now(), 5, 10
	s3clnt, log = s3_helper(), logger('training')

	while True:

		if datetime.now() < next_run:
			time.sleep((next_run-datetime.now()).total_seconds())
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
				train_one_state(click_data, state)
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
