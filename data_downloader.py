from bing_image_downloader import downloader

downloader.download('Virat Kohli', limit=100,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)