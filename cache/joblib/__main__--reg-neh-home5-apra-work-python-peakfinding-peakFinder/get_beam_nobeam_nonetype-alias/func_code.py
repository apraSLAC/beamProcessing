# first line: 231
@mem.cache
def get_beam_nobeam_nonetype(data_source_str, detector_str):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	beam, no_beam, none_type = [], [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			image = image.astype(np.uint8)
			image_prep = preprocess(image, resize=rsz, kernel=kernel_size)
			contour = get_contour(image_prep)
			M = get_image_moments(contour=contour)
			if check_for_beam(M):
				beam.append(nevent)
			else:
				no_beam.append(nevent)
		except (TypeError, AttributeError):
			none_type.append(nevent)
	return beam, no_beam, none_type
