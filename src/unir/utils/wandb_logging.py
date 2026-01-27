import os
import wandb

def init_wandb_training(training_args, run_name):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    # if training_args.wandb_entity is not None:
    #     os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    # if training_args.wandb_project is not None:
    #     os.environ["WANDB_PROJECT"] = training_args.wandb_project
    run = wandb.init(
        project="UniR",#os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        config={
            "training_args": training_args.to_dict(),
        },
        resume="allow",
    )

    print(f"[wandb] running: {run.url}")

    return run