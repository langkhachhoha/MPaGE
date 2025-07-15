from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.method.LLMPFG import MPaGE
from llm4ad.method.LLMPFG import EoHProfiler

# If you want to run the bi_tsp_semo example, uncomment the following line:
from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation as ProblemEvaluation

# If you want to run the bi_tsp_semo example, uncomment the following line:
# from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation as ProblemEvaluation

# If you want to run the bi_cvrp example, uncomment the following line:
# from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation as ProblemEvaluation

# If you want to run the bi_kp example, uncomment the following line:
# from llm4ad.task.optimization.bi_kp import BIKPEvaluation as ProblemEvaluation

# Set your LLM API key here
with open("secret.txt", "r") as f:
    secret = f
    llm_api_key = secret.readline().strip()

with open("secret_cluster.txt", "r") as f:
    secret_cluster = f
    llm_api_key_cluster = secret_cluster.readline().strip()


def main():

    # llm = HttpsApi(host='api.openai.com',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
    #                key=llm_api_key, 
    #                model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
    #                timeout=30
    #                )

    llm = HttpsApiOpenAI(base_url='https://api.openai.com', 
                            api_key=llm_api_key,
                            model='gpt-4o-mini', 
                            timeout=30
                            )
    llm_cluster = HttpsApiOpenAI4Cluster(base_url='https://api.openai.com',
                                        api_key=llm_api_key_cluster,
                                        model='gpt-4o-mini',
                                        timeout=30)
    task = ProblemEvaluation()

    method = MPaGE(llm=llm,
                    llm_cluster=llm_cluster,
                    profiler=EoHProfiler(log_dir='logs', log_style='complex'),
                    evaluation=task,
                    max_sample_nums=200,
                    max_generations=20,
                    pop_size=6,
                    num_samplers=1,
                    num_evaluators=1,
                    #  llm_review=True
                 )

    method.run()


if __name__ == '__main__':
    main()


