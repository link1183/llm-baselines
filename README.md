# LLM Submission - LauzHax (team 5)

## Installation and running

Make sure to install the dependencies, using the following command.

```shell
python -m pip install -r requirements.txt
```

You can then run the model, with our parameters, using the following command.

```shell
python src/main.py --wandb --model llama2 --lr 1e-3 --n_layer 10 --batch_size 30 --grad_clip 1.0 --iterations 23000
```