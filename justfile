default:
  just --list

edit notebook:
  open -a Marimo.app
  sudo marimo edit {{notebook}} --headless --no-token
