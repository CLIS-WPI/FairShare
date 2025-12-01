"""
Learned fairness metrics using autoencoder and GNN.

Uses deep learning to learn fairness representations from
allocation patterns and user performance data.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Learned metrics will use fallback methods.")


@dataclass
class AllocationProfile:
    """Allocation profile for a user/operator."""
    user_id: str
    operator_id: str
    
    # Allocation features
    allocated_bandwidth_mhz: float
    allocated_frequency_mhz: float
    allocation_duration_s: float
    
    # Performance features
    throughput_mbps: float
    latency_ms: float
    packet_loss_rate: float
    
    # Context features
    user_demand_mbps: float
    priority: float
    coverage_quality: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.allocated_bandwidth_mhz / 1000.0,  # Normalize
            self.allocated_frequency_mhz / 40000.0,  # Normalize
            self.allocation_duration_s / 3600.0,  # Normalize to hours
            self.throughput_mbps / 100.0,  # Normalize
            self.latency_ms / 100.0,  # Normalize
            self.packet_loss_rate,
            self.user_demand_mbps / 100.0,  # Normalize
            self.priority,
            self.coverage_quality
        ])


class FairnessAutoencoder(nn.Module):
    """
    Autoencoder for learning fairness representations.
    
    Encodes allocation profiles into a latent space where
    similar fairness patterns are clustered together.
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 4,
        hidden_dims: List[int] = [64, 32]
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """Encode to latent space."""
        return self.encoder(x)


class LearnedFairness:
    """
    Learned fairness metrics using deep learning.
    
    Uses autoencoder to learn fairness representations and
    GNN (optional) for graph-based fairness analysis.
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        use_gpu: bool = True
    ):
        """
        Initialize learned fairness calculator.
        
        Args:
            latent_dim: Latent space dimension
            use_gpu: Whether to use GPU
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for learned fairness metrics")
        
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Autoencoder model
        self.autoencoder = FairnessAutoencoder(latent_dim=latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training state
        self.is_trained = False
    
    def train(
        self,
        profiles: List[AllocationProfile],
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train autoencoder on allocation profiles.
        
        Args:
            profiles: List of allocation profiles
            epochs: Number of training epochs
            batch_size: Batch size
        """
        if not profiles:
            return
        
        # Convert to tensors
        X = np.array([p.to_vector() for p in profiles])
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(X))
            X_shuffled = X_tensor[indices]
            
            # Mini-batch training
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i + batch_size]
                
                # Forward
                self.optimizer.zero_grad()
                reconstructed, latent = self.autoencoder(batch)
                loss = self.criterion(reconstructed, batch)
                
                # Backward
                loss.backward()
                self.optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
    
    def compute_fairness_embedding(
        self,
        profiles: List[AllocationProfile]
    ) -> np.ndarray:
        """
        Compute fairness embeddings for profiles.
        
        Args:
            profiles: List of allocation profiles
            
        Returns:
            Embedding matrix [N, latent_dim]
        """
        if not self.is_trained:
            raise ValueError("Autoencoder not trained. Call train() first.")
        
        if not profiles:
            return np.array([])
        
        # Convert to tensor
        X = np.array([p.to_vector() for p in profiles])
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Encode
        self.autoencoder.eval()
        with torch.no_grad():
            embeddings = self.autoencoder.encode(X_tensor)
        
        return embeddings.cpu().numpy()
    
    def compute_fairness_score(
        self,
        profiles: List[AllocationProfile]
    ) -> float:
        """
        Compute fairness score based on embedding diversity.
        
        More diverse embeddings = more fair (different users get different
        but appropriate allocations).
        
        Args:
            profiles: List of allocation profiles
            
        Returns:
            Fairness score (0.0 to 1.0)
        """
        if not profiles:
            return 0.0
        
        embeddings = self.compute_fairness_embedding(profiles)
        
        if len(embeddings) == 0:
            return 0.0
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        
        distances = pdist(embeddings, metric='euclidean')
        
        # Fairness: higher diversity (higher distances) = more fair
        # But we want reasonable diversity, not extreme
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Normalize to [0, 1]
        # Higher mean distance with low std = fair (diverse but consistent)
        fairness = min(1.0, mean_distance / (std_distance + 1e-6))
        
        return float(fairness)
    
    def cluster_fairness_groups(
        self,
        profiles: List[AllocationProfile],
        n_clusters: int = 3
    ) -> Dict[int, List[str]]:
        """
        Cluster profiles into fairness groups.
        
        Args:
            profiles: List of allocation profiles
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster_id to list of user_ids
        """
        if not profiles:
            return {}
        
        embeddings = self.compute_fairness_embedding(profiles)
        
        if len(embeddings) == 0:
            return {}
        
        # K-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        clusters = {}
        for i, profile in enumerate(profiles):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(profile.user_id)
        
        return clusters


class LearnedFairnessFallback:
    """
    Fallback implementation when PyTorch is not available.
    
    Uses simpler statistical methods to approximate learned fairness.
    """
    
    def compute_fairness_score(
        self,
        profiles: List[AllocationProfile]
    ) -> float:
        """Compute fairness using statistical methods."""
        if not profiles:
            return 0.0
        
        # Extract key features
        throughputs = [p.throughput_mbps for p in profiles]
        latencies = [p.latency_ms for p in profiles]
        allocations = [p.allocated_bandwidth_mhz for p in profiles]
        
        # Compute coefficient of variation (lower = more fair)
        from src.fairness.traditional import TraditionalFairness
        
        jain_throughput = TraditionalFairness.jain_index(throughputs)
        jain_allocation = TraditionalFairness.jain_index(allocations)
        
        # Combined fairness
        fairness = (jain_throughput + jain_allocation) / 2.0
        
        return fairness

