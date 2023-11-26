export ILABSERVER=$1
original_dir=$(pwd)
if [ "$2" = "setup" ]; then
	echo "running setup"
	cd setup
	rm setup.log
	srun --nodelist=$ILABSERVER -o setup.log setup_repos.sh
	echo "setup completed"
	cd "$original_dir"
elif [ "$2" = "build" ]; then
	echo "running build"
	cd build
	rm build.log
	export PATH=/usr/local/cuda/bin:$PATH
	srun --nodelist=$ILABSERVER -G 1 -o build.log build_llamacpp.sh
	echo "build completed"
	cd "$original_dir"
elif [ "$2" = "inference" ]; then
	echo "running inference server"
	cd inference
	rm inference.log
	sbatch --nodelist=$ILABSERVER inference_server.sh
	echo "inference server is set up"
fi
