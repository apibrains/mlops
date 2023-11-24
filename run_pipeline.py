from pipelines.training_pipeline import training_pipline
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipline(data_path = "/home/templification/Desktop/Satyam Mishra/MLOps/data/olist_customers_dataset.csv")


    
    