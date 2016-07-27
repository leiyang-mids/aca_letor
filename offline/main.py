from datetime import datetime, timedelta
from get_click_data import *
from train_one_state import *
from simulate_clicks import *
from logger import *
import traceback, time, sys
import numpy as np

from simulate_clicks import *

def main():
	'''
	'''
	next_run, hour, minute = datetime.now(), 5, 10
	s3clnt, log = s3_helper(), logger('training')
	test = False

	while True:

		if datetime.now() < next_run:
			# time.sleep((next_run-datetime.now()).total_seconds())
			continue
		# get click-through data
		try:
			log.start()
			log.trace('get query and click data')
			# click_data = get_click_data() if not test else simulate_clicks()
			click_data = simulate_clicks()
		except Exception as ex:
			traceback.print_exc(file=log.log_handler())
			log.trace('error getting click data, retry in %d minutes.' %minute)
			next_run = datetime.now() + timedelta(minutes=minute)
			continue
		# train for each state
		for state in np.unique(click_data['state']):
			try:
				train_one_state(click_data, state, log)
			except KeyboardInterrupt:
				sys.exit('User termination')
			except Exception as ex:
				traceback.print_exc(file=log.log_handler())
				log.trace('training has encountered an error for state %s' %state)
		# training completed, get next run time
		next_run = datetime.now() + timedelta(hours=hour)
		log.trace('training has completed, next run time is %s' %str(next_run))
		log.stop()
		# upload log file to S3
		s3clnt.upload2(log.log_file(), 'log/'+log.log_file())

if __name__ == "__main__":
	main()
