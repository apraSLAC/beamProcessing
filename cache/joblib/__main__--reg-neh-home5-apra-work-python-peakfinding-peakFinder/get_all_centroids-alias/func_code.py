# first line: 242
@mem.cache
def get_all_centroids(data_source_str, detector_str):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	all_x, all_y = [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			centroid, _ = find_beam(image)
			all_x.append(centroid[0])
			all_y.append(centroid[1])
		except (TypeError, AttributeError):
			all_x.append(None)
			all_y.append(None)
		if nevent > 1300: break
	return all_x, all_y
