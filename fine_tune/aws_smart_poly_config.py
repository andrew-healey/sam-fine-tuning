from .train import CustomSAMTrainer


def main():

    trainer = CustomSAMTrainer()
    trainer.monitored_train()

if __name__ == "__main__":
    print("starting")
    import sys
    args = sys.argv
    print("args",args)

    main()

    # exit
    import sys
    sys.exit(0)