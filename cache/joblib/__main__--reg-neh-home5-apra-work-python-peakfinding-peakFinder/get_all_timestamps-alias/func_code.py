# first line: 211
@mem.cache
def get_all_timestamps(data_source_str, detector_str):
	"""
	Returns a list containing a tuple of the seconds, nanoseconds, and fiducials
	for every event.
	"""
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	timestamps = []
	for event in tqdm(ds.events()):
		evtId = event.get(psana.EventId)
		timestamps.append((evtId.time()[0], evtId.time()[1], evtId.fiducials()))
	return timestamps
