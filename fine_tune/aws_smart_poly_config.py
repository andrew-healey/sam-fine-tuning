from .train import CustomSAMTrainer

def main():
    trainer = CustomSAMTrainer()
    trainer.monitored_train()

if __name__ == "__main__":
    main()