# first line: 215
@mem.cache
def classify_and_plot_events(data_source_str, detector_str, resize=rsz_default):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	all_x = all_y = []
	beam, no_beam, none_type = [], [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			image = to_uint8(image)
			image_prep = preprocess(image, resize=resize)
			contour = get_contour(image_prep)
			M = get_image_moments(contour=contour)
			if check_for_beam(M):
				centroid = [pos//resize for pos in get_centroid(M)]
				beam.append(nevent)
				all_x.append(centroid[0])
				all_y.append(centroid[1])
			else:
				no_beam.append(nevent)
				all_x.append(None)
				all_y.append(None)
		except (TypeError, AttributeError):
			none_type.append(nevent)
			all_x.append(None)
			all_y.append(None)
	return beam, no_beam, none_type, all_x, all_y
