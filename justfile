default:
  just --list

# launch marimo.app to edit a notebook file
edit notebook:
  open -a Marimo.app --background
  marimo edit {{notebook}} --headless --no-token

# start training run
train:
  open -g -a Orbstack
  OVERMIND_PROCESSES=training,logging overmind s -D
  overmind echo

# serve on etherpad
prototype:
  open -g -a Orbstack
  OVERMIND_PROCESSES=serving,logging,ep_peopen overmind s -D
  overmind echo


# profile training run
profile:
  # delete caches
  sudo rm -rf local_cache
  find . -type d -name '__pycache__' -exec rm -rf {} +; find . -type d -name 'local_cache' -exec rm -rf {} +
  rm -rf ~/.dspy_cache/

  # start application
  open -g -a Orbstack
  OVERMIND_PROCESSES=profiling,logging overmind s -D
  overmind echo


# evaluate model on train and val
evaluate:
  open -g -a Orbstack
  OVERMIND_PROCESSES=evaluating,logging overmind s -D
  overmind echo


improve_factual_consistency:
  open -g -a Orbstack
  OVERMIND_PROCESSES=improving_factual_consistency,logging overmind s -D
  overmind echo
