from .train import CustomSAMTrainer

# use .env file
from dotenv import load_dotenv
load_dotenv()

def main():
    trainer = CustomSAMTrainer()
    trainer.monitored_train()

if __name__ == "__main__":
    main()

    # exit
    import sys
    sys.exit(0)