export ILABSERVER=$1
if [ "$2" = "setup" ]; then
	echo "running setup"
	cd setup
	rm setup.log
	ssh $ILABSERVER /common/home/$USER/Persona-LLM-Chatbot-For-QA/llm_hosting/base_llm/setup/setup_repos.sh > setup.log 2>&1 
	echo "setup completed"
elif [ "$2" = "build" ]; then
	echo "running build"
	cd build
	rm build.log
	export PATH=/usr/local/cuda/bin:$PATH
	srun --nodelist=$ILABSERVER -G 1 -o build.log build_llamacpp.sh
	echo "build completed"
elif [ "$2" = "inference" ]; then
	echo "running inference server"
	cd inference
	rm inference.log
	sbatch --nodelist=$ILABSERVER inference_server.sh
	echo "inference server is set up"
elif [ "$2" = "oai_api" ]; then
	echo "running OPENAI like API"
	cd inference
	rm api.log
	ssh $ILABSERVER /common/home/$USER/Persona-LLM-Chatbot-For-QA/llm_hosting/base_llm/inference/oai_like_api.sh > api.log 2>&1 &
	echo "OPENAI like API is set up"
fi
