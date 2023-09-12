from .train import CustomSAMTrainer


def main():

    trainer = CustomSAMTrainer()
    trainer.monitored_train()

if __name__ == "__main__":
    main()

    # exit
    import sys
    sys.exit(0)