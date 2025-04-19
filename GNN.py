import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import BatchNorm1d
import time
import logging
from datetime import datetime

# Configure logging
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/gnn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('GNN_Training')

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MoleculeFeaturizer:
    """Class for converting SMILES to molecular graphs with atom features."""
    
    @staticmethod
    def smiles_to_graph(smiles):
        """Convert a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Extract atom features
        num_atoms = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            feature = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetImplicitValence(),
                atom.GetIsAromatic() * 1, 
                atom.GetHybridization().real
            ]
            features.append(feature)
        
        x = torch.tensor(features, dtype=torch.float)
        
        # Extract edge indices
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Add both directions for undirected graph
        
        if len(edge_indices) == 0:  # Handle molecules with no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    @staticmethod
    def augment_smiles(smiles, n_augments=5):
        """Generate n_augments random SMILES for a given SMILES input."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        smiles_set = set()
        for _ in range(n_augments):
            new_smiles = Chem.MolToSmiles(mol, doRandom=True)
            smiles_set.add(new_smiles)

        return list(smiles_set)

    @staticmethod
    def smiles_regression_augmentation(original_df, n_augments=5):
        """
        Augment a DataFrame with random SMILES representations.
        
        Args:
            original_df: pandas DataFrame with 'SMILES' and 'pIC50' columns
            n_augments: number of augmentations per molecule
            
        Returns:
            DataFrame with augmented SMILES and corresponding pIC50 values
        """
        augmented_data = []

        for i, row in original_df.iterrows():
            original_smiles = row['SMILES']
            pic50 = row['pIC50']
            
            augmented_data.append((original_smiles, pic50))
            
            aug_smiles_list = MoleculeFeaturizer.augment_smiles(original_smiles, n_augments=n_augments)
            for aug_smiles in aug_smiles_list:
                augmented_data.append((aug_smiles, pic50))

        aug_df = pd.DataFrame(augmented_data, columns=['SMILES', 'pIC50'])
        return aug_df


class MoleculeDataset(Dataset):
    """PyTorch Geometric Dataset for molecular graph data."""
    
    def __init__(self, smiles_list, labels):
        super(MoleculeDataset, self).__init__()
        self.smiles_list = smiles_list
        self.labels = labels
        
        self.processed_data = []
        for i, smiles in enumerate(smiles_list):
            graph = MoleculeFeaturizer.smiles_to_graph(smiles)
            if graph is not None:
                graph.y = torch.tensor([labels[i]], dtype=torch.float)
                self.processed_data.append(graph)
    
    def len(self):
        return len(self.processed_data)
    
    def get(self, idx):
        return self.processed_data[idx]


class GNN(nn.Module):
    """Graph Neural Network model for molecular property prediction."""
    
    def __init__(self, num_features, hidden_channels, dropout=0.1):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x.view(-1)


class ModelTrainer:
    """Class for training and evaluating GNN models."""
    
    def __init__(self, model, device, model_save_path='models/best_gnn_model.pt'):
        self.model = model
        self.device = device
        self.model_save_path = model_save_path
        
    def train(self, train_loader, optimizer):
        """Train the model for one epoch."""
        self.model.train()
        loss_all = 0
        
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.mse_loss(output, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
        
        return loss_all / len(train_loader.dataset)

    def evaluate(self, loader):
        """Evaluate the model on the validation or test set."""
        self.model.eval()
        predictions = []
        actual = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data.x, data.edge_index, data.batch)
                predictions.append(pred)
                actual.append(data.y)
        
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        actual = torch.cat(actual, dim=0).cpu().numpy()
        
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actual': actual
        }
    
    def train_model(self, train_loader, test_loader, epochs=200, patience=15, lr=0.01):
        """Train the model with early stopping."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        logger.info('Starting training...')
        best_mse = float('inf')
        best_epoch = 0
        no_improve_count = 0
        history = {
            'train_loss': [],
            'test_mse': [],
            'test_rmse': [],
            'test_mae': [],
            'test_r2': []
        }
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train(train_loader, optimizer)
            eval_metrics = self.evaluate(test_loader)
            
            history['train_loss'].append(train_loss)
            history['test_mse'].append(eval_metrics['mse'])
            history['test_rmse'].append(eval_metrics['rmse'])
            history['test_mae'].append(eval_metrics['mae'])
            history['test_r2'].append(eval_metrics['r2'])
            
            scheduler.step(eval_metrics['mse'])
            
            if eval_metrics['mse'] < best_mse:
                best_mse = eval_metrics['mse']
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.model_save_path)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if epoch % 10 == 0:
                logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                          f'Test MSE: {eval_metrics["mse"]:.4f}, Test R²: {eval_metrics["r2"]:.4f}')
            
            # Early stopping
            if no_improve_count >= patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
        logger.info(f'Best model at epoch {best_epoch} with MSE {best_mse:.4f}')
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(self.model_save_path))
        return history

    def k_fold_cv(self, dataset, k=5, batch_size=16, epochs=200, lr=0.01):
        """Perform k-fold cross-validation."""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        cv_results = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        fold_indices = list(kf.split(range(len(dataset))))
        
        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            logger.info(f"Starting fold {fold+1}/{k}")
            
            # Reset model weights
            self.model.apply(self._weight_reset)
            
            train_loader = DataLoader(
                [dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(
                [dataset[i] for i in test_idx], batch_size=batch_size
            )
            
            # Train model
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            best_mse = float('inf')
            for epoch in range(1, epochs + 1):
                train_loss = self.train(train_loader, optimizer)
                eval_metrics = self.evaluate(test_loader)
                scheduler.step(eval_metrics['mse'])
                
                if eval_metrics['mse'] < best_mse:
                    best_mse = eval_metrics['mse']
                    torch.save(self.model.state_dict(), f"models/fold_{fold+1}_best_model.pt")
                
                if epoch % 50 == 0:
                    logger.info(f'Fold {fold+1}, Epoch: {epoch}, MSE: {eval_metrics["mse"]:.4f}, R²: {eval_metrics["r2"]:.4f}')
            
            # Load best model for this fold
            self.model.load_state_dict(torch.load(f"models/fold_{fold+1}_best_model.pt"))
            final_metrics = self.evaluate(test_loader)
            
            cv_results['mse'].append(final_metrics['mse'])
            cv_results['rmse'].append(final_metrics['rmse'])
            cv_results['mae'].append(final_metrics['mae'])
            cv_results['r2'].append(final_metrics['r2'])
            
            logger.info(f"Fold {fold+1} results: MSE={final_metrics['mse']:.4f}, R²={final_metrics['r2']:.4f}")
        
        # Calculate average metrics
        avg_results = {metric: np.mean(values) for metric, values in cv_results.items() if metric != 'predictions' and metric != 'actual'}
        std_results = {f"{metric}_std": np.std(values) for metric, values in cv_results.items() if metric != 'predictions' and metric != 'actual'}
        
        combined_results = {**avg_results, **std_results}
        
        for metric, value in combined_results.items():
            logger.info(f"Average {metric}: {value:.4f}")
        
        return combined_results
    
    @staticmethod
    def _weight_reset(m):
        """Reset model weights."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, GATConv) or isinstance(m, GCNConv):
            m.reset_parameters()


class Visualizer:
    """Class for visualizing model results."""
    
    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred, save_path='reports/prediction_scatter.png'):
        """Plot actual vs predicted values with regression metrics."""
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
        
        # Plot the perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Add trend line
        z = np.polyfit(y_true.flatten(), y_pred.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.sort(y_true.flatten()), p(np.sort(y_true.flatten())), 
                 'b-', alpha=0.7, label=f'Trend line (y = {z[0]:.2f}x + {z[1]:.2f})')
        
        plt.xlabel('Actual pIC50', fontsize=14)
        plt.ylabel('Predicted pIC50', fontsize=14)
        plt.title(f'Actual vs Predicted pIC50\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Scatter plot saved to {save_path}")
        return plt
    
    @staticmethod
    def plot_training_curves(history, save_path='reports/training_curves.png'):
        """Plot training and validation metrics over epochs."""
        plt.figure(figsize=(15, 10))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'b-', label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training Loss Over Epochs', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test MSE
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(history['test_mse']) + 1), history['test_mse'], 'r-', label='Test MSE')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title('Test MSE Over Epochs', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test R²
        plt.subplot(2, 2, 3)
        plt.plot(range(1, len(history['test_r2']) + 1), history['test_r2'], 'g-', label='Test R²')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Test R² Over Epochs', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot test RMSE
        plt.subplot(2, 2, 4)
        plt.plot(range(1, len(history['test_rmse']) + 1), history['test_rmse'], 'm-', label='Test RMSE')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Test RMSE Over Epochs', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Training curves saved to {save_path}")
        return plt
    
    @staticmethod
    def generate_prediction_report(test_results, save_path='reports/prediction_report.csv'):
        """Generate a CSV report with actual and predicted values."""
        results_df = pd.DataFrame({
            'Actual_pIC50': test_results['actual'].flatten(),
            'Predicted_pIC50': test_results['predictions'].flatten(),
            'Absolute_Error': np.abs(test_results['actual'].flatten() - test_results['predictions'].flatten())
        })
        results_df.to_csv(save_path, index=False)
        logger.info(f"Prediction report saved to {save_path}")
        return results_df
    
    @staticmethod
    def plot_error_distribution(y_true, y_pred, save_path='reports/error_distribution.png'):
        """Plot the distribution of prediction errors."""
        errors = y_pred - y_true
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Errors', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # QQ plot
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(errors.flatten(), dist="norm", plot=plt)
        plt.title('Q-Q Plot of Prediction Errors', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Error distribution plot saved to {save_path}")
        return plt


def build_and_train_gnn_model(smiles_list, pic50_values, test_size=0.2, epochs=200, 
                              batch_size=16, hidden_channels=64, dropout=0.1,
                              use_augmentation=False, n_augments=5, run_cv=False, device=None):
    """
    Build and train a GNN model for molecular property prediction.
    
    Args:
        smiles_list: List of SMILES strings
        pic50_values: List of pIC50 values
        test_size: Fraction of data to use for testing
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        hidden_channels: Number of hidden channels in GNN
        dropout: Dropout probability
        use_augmentation: Whether to use SMILES augmentation
        n_augments: Number of augmentations per molecule
        run_cv: Whether to run cross-validation
        
    Returns:
        model: Trained GNN model
        results: Dictionary of evaluation metrics
    """
    set_seed(42)
    start_time = time.time()
    
    # Create directory for results if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logger.info("Preparing dataset...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Test if CUDA is actually working
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear cache first
            test_tensor = torch.tensor([1.0], device=device)
    except RuntimeError:
        logger.warning("CUDA error encountered. Falling back to CPU.")
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")

    # Augment data if requested
    if use_augmentation:
        logger.info(f"Augmenting SMILES with {n_augments} variants per molecule...")
        original_df = pd.DataFrame({'SMILES': smiles_list, 'pIC50': pic50_values})
        augmented_df = MoleculeFeaturizer.smiles_regression_augmentation(original_df, n_augments)
        
        smiles_list = augmented_df['SMILES'].tolist()
        pic50_values = augmented_df['pIC50'].tolist()
        
        logger.info(f"Dataset size after augmentation: {len(smiles_list)} molecules")
    
    # Create dataset
    dataset = MoleculeDataset(smiles_list, pic50_values)
    logger.info(f"Created dataset with {len(dataset)} valid molecules")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = GNN(num_features=6, hidden_channels=hidden_channels, dropout=dropout).to(device)
    logger.info(f"Created GNN model with {hidden_channels} hidden channels and {dropout} dropout")
    
    # Create trainer
    trainer = ModelTrainer(model, device)
    
    if run_cv:
        logger.info("Running 5-fold cross-validation...")
        cv_results = trainer.k_fold_cv(dataset, k=5, batch_size=batch_size, epochs=epochs)
        
        # Save CV results
        cv_df = pd.DataFrame([cv_results])
        cv_df.to_csv('reports/cv_results.csv', index=False)
        logger.info(f"CV results saved to reports/cv_results.csv")
        
        final_results = cv_results
    else:
        # Split dataset into train and test
        train_indices, test_indices = train_test_split(
            list(range(len(dataset))), test_size=test_size, random_state=42
        )
        
        train_loader = DataLoader(
            [dataset[i] for i in train_indices], batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_indices], batch_size=batch_size
        )
        
        logger.info(f"Training with {len(train_indices)} samples, testing with {len(test_indices)} samples")
        
        # Train model
        history = trainer.train_model(train_loader, test_loader, epochs=epochs)
        
        # Plot training curves
        Visualizer.plot_training_curves(history)
        
        # Final evaluation
        test_results = trainer.evaluate(test_loader)
        
        # Generate visualizations
        Visualizer.plot_actual_vs_predicted(
            test_results['actual'], 
            test_results['predictions'],
            save_path='reports/prediction_scatter.png'
        )
        
        Visualizer.plot_error_distribution(
            test_results['actual'], 
            test_results['predictions'],
            save_path='reports/error_distribution.png'
        )
        
        # Generate prediction report
        Visualizer.generate_prediction_report(test_results)
        
        # Log results
        logger.info(f"Final results on test set:")
        logger.info(f"MSE: {test_results['mse']:.4f}")
        logger.info(f"RMSE: {test_results['rmse']:.4f}")
        logger.info(f"MAE: {test_results['mae']:.4f}")
        logger.info(f"R²: {test_results['r2']:.4f}")
        
        final_results = test_results
    
    end_time = time.time()
    logger.info(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes")
    
    return model, final_results


def predict_new_molecules(model, smiles_list, device=None):
    """
    Make predictions for new molecules.
    
    Args:
        model: Trained GNN model
        smiles_list: List of SMILES strings for prediction
        device: Device to use for predictions
        
    Returns:
        DataFrame with SMILES and predicted pIC50 values
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    predictions = []
    valid_smiles = []
    
    for smiles in smiles_list:
        graph = MoleculeFeaturizer.smiles_to_graph(smiles)
        if graph is not None:
            # Add batch dimension (just one molecule)
            graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
            graph = graph.to(device)
            
            with torch.no_grad():
                pred = model(graph.x, graph.edge_index, graph.batch).item()
            
            predictions.append(pred)
            valid_smiles.append(smiles)
    
    results_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Predicted_pIC50': predictions
    })
    
    return results_df


def main():
    """Main function to run the GNN model training and evaluation."""
    parser = argparse.ArgumentParser(description='Train GNN model for molecular property prediction')
    parser.add_argument('--data', type=str, default='xanthone_derivatives.csv', 
                        help='Path to input CSV file with SMILES and pIC50 columns')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in GNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--augment', action='store_true', help='Use SMILES augmentation')
    parser.add_argument('--n_augments', type=int, default=5, help='Number of augmentations per molecule')
    parser.add_argument('--cross_validate', action='store_true', help='Run 5-fold cross-validation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    try:
        df = pd.read_csv(args.data)
        smiles_list = df['SMILES'].tolist()
        pic50_values = df['pIC50'].tolist()
        logger.info(f"Loaded {len(smiles_list)} molecules")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    if args.cpu:
        device = torch.device('cpu')
    else:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except RuntimeError:
            device = torch.device('cpu')
    # Build and train model
    model, results = build_and_train_gnn_model(
    smiles_list=smiles_list,
    pic50_values=pic50_values,
    test_size=args.test_size,
    epochs=args.epochs,
    batch_size=args.batch_size,
    hidden_channels=args.hidden_channels,
    dropout=args.dropout,
    use_augmentation=args.augment,
    n_augments=args.n_augments,
    run_cv=args.cross_validate,
    device=device 
    )
    
    if not args.cross_validate:
        logger.info(f"Final results: MSE = {results['mse']:.4f}, R² = {results['r2']:.4f}")
    
    
            
if __name__ == "__main__":
    main()