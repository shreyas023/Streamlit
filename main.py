import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import streamlit as st
from sklearn.model_selection import train_test_split

# Import your custom classes and functions
from model import DataPreprocessor, TimeSeriesDataset, BiRNNImputer, train_model, impute_data

def main():
    st.title("DataSynth: Time Series Imputation")
    st.sidebar.header("Configuration Panel")

    # File upload
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    if uploaded_file:
        try:
            df = pd.read_json(uploaded_file)
            st.write("Dataset Loaded Successfully!")
            st.write(df.head())
            st.session_state["df"] = df  # Store the dataframe in session state
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # Parameters
    sequence_length = st.sidebar.slider("Sequence Length", 5, 50, 10)
    hidden_size = st.sidebar.slider("Hidden Size", 32, 256, 64)
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 100, 50)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.write(f"Device: {device}")

    if st.button("Preprocess Data"):
        if "df" in st.session_state:
            df = st.session_state["df"]

            # Drop the 'timestamp' column if it exists
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
                st.write("Dropped 'timestamp' column.")

            # Drop non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]
            
            # Check for remaining non-numeric columns
            if not df.select_dtypes(exclude=[np.number]).empty:
                st.error("Non-numeric columns detected. Please ensure the dataset contains only numeric columns.")
                return

            preprocessor = DataPreprocessor()
            preprocessor.fit(df)

            data_transformed = preprocessor.transform(df)
            st.write("Preprocessed Data")
            st.write(pd.DataFrame(data_transformed).head())

            dataset = TimeSeriesDataset(data_transformed, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Store in session state
            st.session_state["preprocessor"] = preprocessor
            st.session_state["data_transformed"] = data_transformed
            st.session_state["train_loader"] = train_loader
            st.session_state["val_loader"] = val_loader
            st.success("Data Preprocessing Completed!")
        else:
            st.error("Please upload a dataset first!")

        if "df" in st.session_state:
            df = st.session_state["df"]

            # Drop the 'timestamp' column if it exists
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
                st.write("Dropped 'timestamp' column.")

            # Drop non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_columns]

            preprocessor = DataPreprocessor()
            preprocessor.fit(df)

            data_transformed = preprocessor.transform(df)
            st.write("Preprocessed Data")
            st.write(pd.DataFrame(data_transformed).head())

            dataset = TimeSeriesDataset(data_transformed, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Store in session state
            st.session_state["preprocessor"] = preprocessor
            st.session_state["data_transformed"] = data_transformed
            st.session_state["train_loader"] = train_loader
            st.session_state["val_loader"] = val_loader
            st.success("Data Preprocessing Completed!")
        else:
            st.error("Please upload a dataset first!")

    if st.button("Train Model"):
        if "train_loader" in st.session_state and "val_loader" in st.session_state:
            data_transformed = st.session_state["data_transformed"]
            model = BiRNNImputer(input_size=data_transformed.shape[1],
                                 hidden_size=hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_losses = train_model(
                model,
                st.session_state["train_loader"],
                st.session_state["val_loader"],
                st.session_state["preprocessor"],
                optimizer,
                num_epochs,
                device,
            )

            st.success("Model Training Completed!")
            st.line_chart({"Train Loss": train_losses, "Validation Loss": val_losses})
            st.session_state["model"] = model
        else:
            st.error("Please preprocess the data first!")

    if st.button("Impute Data"):
        if "model" in st.session_state:
            data_transformed = st.session_state["data_transformed"]
            imputed_df = impute_data(
                st.session_state["model"],
                data_transformed,
                sequence_length,
                st.session_state["preprocessor"],
                st.session_state["df"],
                device,
            )
            st.write("Imputed Data")
            st.write(imputed_df)
            st.download_button("Download Imputed Data", imputed_df.to_csv(index=False), "imputed_data.csv")
        else:
            st.error("Train the model before imputing data!")


if __name__ == "__main__":
    main()
