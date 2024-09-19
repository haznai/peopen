default:
  just --list

# launch marimo.app to edit a notebook file
edit notebook:
  open -a Marimo.app --background
  marimo edit {{notebook}} --headless --no-token

# start training run
train:
  overmind s -D
