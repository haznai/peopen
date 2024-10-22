logging: python -m phoenix.server.main serve
training: sleep 1 && python src/model_training.py
profiling: sleep 1 && scalene src/model_training.py
serving: sleep 1 && python src/model_serving.py
evaluating: sleep 1 && python src/model_evaluation.py
ep_peopen: env -C ../ep_peopen just rebuild
improving_factual_consistency: sleep 1 && python src/submodels/improve_factual_consistency_model.py
