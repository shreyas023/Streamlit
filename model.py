import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DataPreprocessor:
    def __init__(self):
        self.numerical_columns = []
        self.categorical_columns = []
        self.label_encoders = {}
        self.numerical_scaler = StandardScaler()

    def fit(self, df):
        # Identify numerical and categorical columns
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Exclude datetime columns like 'timestamp'
        datetime_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        self.numerical_columns = [col for col in self.numerical_columns if col not in datetime_columns]

        # Create numerical column indices
        self.numerical_indices = [i for i, col in enumerate(df.columns)
                                if col in self.numerical_columns]
        self.categorical_indices = [i for i, col in enumerate(df.columns)
                                  if col in self.categorical_columns]

        # Fit numerical scaler
        if len(self.numerical_columns) > 0:
            self.numerical_scaler.fit(df[self.numerical_columns].fillna(df[self.numerical_columns].mean()))

        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            filled_col = df[col].fillna(df[col].mode()[0])
            self.label_encoders[col].fit(filled_col)

    def transform(self, df):
        df_transformed = df.copy()

        # Exclude datetime columns
        datetime_columns = df_transformed.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        df_transformed.drop(columns=datetime_columns, inplace=True)

        # Transform numerical columns
        if self.numerical_columns:
            available_numerical_columns = [col for col in self.numerical_columns if col in df_transformed.columns]
            if available_numerical_columns:
                df_transformed[available_numerical_columns] = self.numerical_scaler.transform(
                    df_transformed[available_numerical_columns].fillna(df_transformed[available_numerical_columns].mean())
                )

        # Transform categorical columns
        for col in self.categorical_columns:
            if col in df_transformed.columns:
                filled_col = df_transformed[col].fillna(df_transformed[col].mode()[0])
                df_transformed[col] = self.label_encoders[col].transform(filled_col)

        # Return transformed values as float32, excluding any dropped columns
        return df_transformed.values.astype(np.float32)

    def inverse_transform(self, data, original_df):
        """
        Convert the transformed data back to a DataFrame format with the same columns as the original DataFrame.
        """
        # Create a DataFrame with the transformed data
        reconstructed_df = pd.DataFrame(data, columns=self.numerical_columns + self.categorical_columns)
        
        # Inverse transform numerical columns
        if self.numerical_columns:
            numerical_data = reconstructed_df[self.numerical_columns].values
            # Reshape the data to 2D array if needed
            if numerical_data.ndim == 1:
                numerical_data = numerical_data.reshape(-1, len(self.numerical_columns))
            
            # Perform inverse transform
            reconstructed_values = self.numerical_scaler.inverse_transform(numerical_data)
            reconstructed_df[self.numerical_columns] = reconstructed_values

        # Inverse transform categorical columns
        for col in self.categorical_columns:
            reconstructed_df[col] = self.label_encoders[col].inverse_transform(
                reconstructed_df[col].astype(int)
            )

        # Restore any columns that were in the original DataFrame but not transformed
        for col in original_df.columns:
            if col not in reconstructed_df.columns:
                reconstructed_df[col] = original_df[col]

        # Ensure column order matches original DataFrame
        return reconstructed_df[original_df.columns]


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.sequence_length]


def custom_loss(outputs, targets, numerical_indices, categorical_indices, preprocessor):
    """Custom loss function that handles both numerical and categorical variables"""
    loss = 0

    # Get the actual shapes
    if outputs.dim() == 3:  # If shape is [batch_size, sequence_length, features]
        batch_size, seq_len, num_features = outputs.size()
        outputs = outputs.reshape(-1, num_features)  # Reshape to [batch_size * seq_len, features]
        targets = targets.reshape(-1, num_features)

    # MSE for numerical variables
    if numerical_indices:
        numerical_outputs = outputs[:, numerical_indices]
        numerical_targets = targets[:, numerical_indices]
        loss += nn.MSELoss()(numerical_outputs, numerical_targets)

    # Cross-entropy for categorical variables
    if categorical_indices:
        for idx in categorical_indices:
            # Get number of unique classes for this categorical variable
            num_classes = len(preprocessor.label_encoders[preprocessor.categorical_columns[
                categorical_indices.index(idx)]].classes_)

            # Create logits for each class
            cat_outputs = outputs[:, idx].view(-1, num_classes)

            # Get target classes
            cat_targets = targets[:, idx].long()

            # Calculate cross-entropy loss
            loss += nn.CrossEntropyLoss()(cat_outputs, cat_targets)

    return loss

def custom_loss(outputs, targets, numerical_indices, categorical_indices, preprocessor):
    """Custom loss function that handles both numerical and categorical variables"""
    loss = 0

    # Flatten dimensions if necessary
    if outputs.dim() == 3:  # If shape is [batch_size, sequence_length, features]
        batch_size, seq_len, num_features = outputs.size()
        outputs = outputs.reshape(-1, num_features)  # Reshape to [batch_size * seq_len, features]
        targets = targets.reshape(-1, num_features)

    # MSE for numerical variables
    if numerical_indices:
        numerical_outputs = outputs[:, numerical_indices]
        numerical_targets = targets[:, numerical_indices]
        loss += F.mse_loss(numerical_outputs, numerical_targets)

    # Cross-entropy for categorical variables
    if categorical_indices:
        for idx in categorical_indices:
            # Retrieve the specific column and label encoder for the categorical feature
            col_name = preprocessor.categorical_columns[categorical_indices.index(idx)]
            num_classes = len(preprocessor.label_encoders[col_name].classes_)

            # Select predictions for the categorical variable and ensure correct shape
            cat_outputs = outputs[:, idx].unsqueeze(1).expand(-1, num_classes)
            cat_targets = targets[:, idx].long().clamp(0, num_classes - 1)

            # Apply CrossEntropyLoss to the categorical feature
            loss += F.cross_entropy(cat_outputs, cat_targets)

    return loss

class BiRNNImputer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(BiRNNImputer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.birnn = nn.RNN(input_size, hidden_size, num_layers,
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, features]
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        out, _ = self.birnn(x, h0)
        out = self.fc(out)  # Shape: [batch_size, sequence_length, features]
        return out

def train_model(model, train_loader, val_loader, preprocessor, optimizer, num_epochs, device):
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            batch = batch.to(device)

            # Create missing value mask (random masking for training)
            mask = torch.rand_like(batch) < 0.15
            masked_input = batch.clone()
            masked_input[mask] = 0

            # Forward pass
            outputs = model(masked_input)

            # Calculate loss
            loss = custom_loss(
                outputs,
                batch,
                preprocessor.numerical_indices,
                preprocessor.categorical_indices,
                preprocessor
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                mask = torch.rand_like(batch) < 0.15
                masked_input = batch.clone()
                masked_input[mask] = 0

                outputs = model(masked_input)
                loss = custom_loss(
                    outputs,
                    batch,
                    preprocessor.numerical_indices,
                    preprocessor.categorical_indices,
                    preprocessor
                )
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')

        model.train()

    return train_losses, val_losses

def impute_data(model, data, sequence_length, preprocessor, original_df, device):
    model.eval()

    # Exclude timestamp or dropped columns
    missing_mask = pd.isna(original_df.drop(columns=['timestamp'], errors='ignore'))

    # Transform data
    data_transformed = preprocessor.transform(original_df.drop(columns=['timestamp'], errors='ignore'))

    # Ensure data shapes are consistent
    if data_transformed.shape[1] != len(preprocessor.numerical_columns) + len(preprocessor.categorical_columns):
        raise ValueError("Mismatch between transformed data and expected column count.")

    data_tensor = torch.FloatTensor(data_transformed).to(device)
    missing_tensor = torch.tensor(missing_mask.values).to(device)

    with torch.no_grad():
        # Initialize imputed data
        imputed_data = data_transformed.copy()

        for i in range(0, len(data_transformed), sequence_length):
            sequence = data_tensor[i:i + sequence_length].unsqueeze(0)
            sequence_missing = missing_tensor[i:i + sequence_length].unsqueeze(0)
            outputs = model(sequence)

            for j in range(sequence.size(1)):
                if i + j < len(data_transformed):
                    missing_indices = torch.where(sequence_missing[0, j])[0]
                    if len(missing_indices) > 0:
                        imputed_data[i + j, missing_indices.cpu().numpy()] = \
                            outputs[0, j, missing_indices].cpu().numpy()

    # Convert back to original DataFrame format
    imputed_df = preprocessor.inverse_transform(imputed_data, original_df.drop(columns=['timestamp'], errors='ignore'))
    return imputed_df

def main():
    try:
        # Load datasets
        df_csv1 = pd.read_json(r"/content/smartwatch_vitals_amputed.json")
        df_csv2 = pd.read_json(r"/content/smartwatch_vitals.json")

        print("Dataset shape:", df_csv1.shape)
        print("\nColumn types:")
        print(df_csv1.dtypes)

        # Parameters
        sequence_length = 10
        hidden_size = 64
        num_epochs = 50
        batch_size = 32
        learning_rate = 0.001
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Before preprocessing, drop or convert 'timestamp'
        # Remove timestamp for all operations
        df_csv1 = df_csv1.copy()
        df_csv1['timestamp'] = pd.to_datetime(df_csv1['timestamp'])
        df_csv1.drop(columns=['timestamp'], inplace=True)

        # Impute missing values
        imputed_df = impute_data(
            model,
            df_csv1,  # Pass the modified DataFrame without timestamp
            sequence_length,
            preprocessor,
            df_csv1,  # Pass the same modified DataFrame for consistency
            device
        )

        # Initialize preprocessor and prepare data
        preprocessor = DataPreprocessor()
        preprocessor.fit(df_csv1)
        data_transformed = preprocessor.transform(df_csv1.drop(columns=['timestamp'], errors='ignore'))

        print("\nPreprocessed data shape:", data_transformed.shape)
        print("Number of numerical features:", len(preprocessor.numerical_indices))
        print("Number of categorical features:", len(preprocessor.categorical_indices))

        dataset = TimeSeriesDataset(data_transformed, sequence_length)

        print("\nDataset size:", len(dataset))
        sample_batch = next(iter(DataLoader(dataset, batch_size=1)))
        print("Sample batch shape:", sample_batch.shape)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        input_size = data_transformed.shape[1]
        model = BiRNNImputer(input_size, hidden_size)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate the model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            preprocessor, optimizer, num_epochs, device
        )

        # Impute missing values
        imputed_df = impute_data(model, df_csv1, sequence_length, preprocessor, df_csv1, device)

        # Save imputed data
        imputed_df.to_json('imputed_data1.json', index=False)

        # Visualize results
        plt.figure(figsize=(15, 10))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_validation_loss.png')
        plt.close()

        print("Imputation completed. Results saved to 'imputed_data.json' and 'training_validation_loss.png'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()