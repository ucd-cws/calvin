# download default model data

import os, requests, zipfile, shutil

# my-models directory
my_model_dir = './my-models'

# data url links
calvin_annual = 'https://www.dropbox.com/s/ac1gxs8y49oiw7d/annual.zip?dl=1'
calvin_pf = 'https://www.dropbox.com/s/ikt5j6kd7n80rir/links82yr.csv.zip?dl=1'
two_res_pilot_lf = 'https://www.dropbox.com/s/4xzmwd23mxtye51/two-sr-pilot-lf.zip?dl=1'

def get_zip_data(url,out):
	# create directory
	if not os.path.isdir(os.path.dirname(out)):
	        os.makedirs(os.path.dirname(out))
	# retrive file
	myfile = requests.get(url)
	open(out, 'wb').write(myfile.content)
	# unzip content
	with zipfile.ZipFile(out,'r') as zip_ref:
	    zip_ref.extractall(os.path.dirname(out))

	os.remove(out)

	try:
	    shutil.rmtree(os.path.join(os.path.dirname(out),'__MACOSX'))
	except OSError:
	    pass


print('Downloading Calvin annual model to: '+my_model_dir,flush=True)
get_zip_data(calvin_annual,os.path.join(my_model_dir,'calvin-annual','annual.zip'))

print('Downloading Calvin perfect foresight model to: '+my_model_dir,flush=True)
get_zip_data(calvin_pf,os.path.join(my_model_dir,'calvin-pf','links82yr.csv.zip'))

print('Downloading two reservoir pilot limited foresight model to: '+my_model_dir,flush=True)
get_zip_data(two_res_pilot_lf,os.path.join(my_model_dir,'two-sr-pilot-lf','two-sr-pilot-lf.zip'))
