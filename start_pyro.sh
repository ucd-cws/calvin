# activate environment that has pyro installed
# unnecesssary if running in base environment
source ~/anaconda3/bin/activate calsim

if  pyro4-nsc ping | grep 'Name server ping ok.'; then

  	echo '(Pyro server is active)'

else

	echo 'Starting pyro server'
	# start pyomo named server 
	pyomo_ns >& ./pyro_logs_pyomo_ns.out &
	echo '...pyro server started'
	sleep 5
	echo 'Starting dispatch server'
	# start dispatch listening server 
	dispatch_srvr >& ./pyro_logs_dispatch_srvr.out &
	echo '...dispatch server started'
	sleep 5
	# start user specified number of mip servers for solving  >& ./temp/pyro_mip_server$i.out &
	for i in `seq 1 $1`;
	 	do
	 		# conda activate calsim
	    	pyro_mip_server >& ./pyro_logs_mip_server$i.out &
	 	done  
fi
