"""
Value Network for EXCHANGE

A convolutional neural network that evaluates board positions.
Output is a single scalar in range [-1, 1]:
- Positive values = good for the side to move
- Negative values = bad for the side to move
- Magnitude indicates confidence

Architecture inspired by AlphaZero but simplified for our smaller problem:
- Residual convolutional blocks for pattern recognition
- Global pooling to aggregate spatial features
- Value head with fully connected layers
- Policy head for move priors (used by MCTS)

The network is intentionally compact (~100k-200k parameters) to ensure
fast inference even on modest hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .game_state import INPUT_CHANNELS, BOARD_SIZE


# Move encoding constants
# Policy output: 64x64 (from_square x to_square) + 64x8 (knight attack directions)
POLICY_FROM_TO_SIZE = 64 * 64  # 4096 - covers MOVE, ATTACK, MOVE_AND_ATTACK destination
POLICY_KNIGHT_ATTACKS = 64 * 8  # 512 - 8 relative attack directions from each square
POLICY_SIZE = POLICY_FROM_TO_SIZE + POLICY_KNIGHT_ATTACKS  # 4608 total


class ResidualBlock(nn.Module):
    """
    A residual block with two conv layers and skip connection.
    Uses batch normalization for training stability.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ExchangeValueNetwork(nn.Module):
    """
    Value network for EXCHANGE position evaluation.

    Architecture:
    1. Input conv: 27 channels -> hidden_channels
    2. N residual blocks (default 4)
    3. Global average pooling
    4. FC layers with value head

    Parameters:
        hidden_channels: Number of channels in conv layers (default 64)
        num_blocks: Number of residual blocks (default 4)
        fc_hidden: Hidden layer size in value head (default 128)
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        fc_hidden: int = 128,
    ):
        super().__init__()

        # Input convolution
        self.input_conv = nn.Conv2d(
            INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, fc_hidden)
        self.value_fc2 = nn.Linear(fc_hidden, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, INPUT_CHANNELS, 8, 8)

        Returns:
            Value tensor of shape (batch, 1) in range [-1, 1]
        """
        # Input processing
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # Flatten: (batch, 64)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Output in [-1, 1]

        return v

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get network configuration for serialization."""
        return {
            "hidden_channels": self.input_conv.out_channels,
            "num_blocks": len(self.res_blocks),
            "fc_hidden": self.value_fc1.out_features,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ExchangeValueNetwork":
        """Create network from config dict."""
        return cls(
            hidden_channels=config.get("hidden_channels", 64),
            num_blocks=config.get("num_blocks", 4),
            fc_hidden=config.get("fc_hidden", 128),
        )


class PolicyValueNetwork(nn.Module):
    """
    Policy-Value network for MCTS-based training.

    Dual-headed architecture:
    - Shared convolutional trunk (same as ExchangeValueNetwork)
    - Value head: Outputs position evaluation in [-1, 1]
    - Policy head: Outputs move prior probabilities

    Policy encoding:
    - 4096 logits for (from_square, to_square) pairs
    - 512 logits for Knight attack directions (8 directions × 64 landing squares)
    - Total: 4608 policy outputs

    Move encoding scheme:
    - MOVE: from*64 + to
    - ATTACK: from*64 + to (attack target)
    - MOVE_AND_ATTACK: from*64 + to (destination) AND 4096 + to*8 + attack_dir
    - ABILITY: from*64 + to (or from*64 + from for stationary abilities)
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        fc_hidden: int = 128,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.fc_hidden = fc_hidden

        # Shared trunk
        self.input_conv = nn.Conv2d(
            INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Value head (same as ExchangeValueNetwork)
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, fc_hidden)
        self.value_fc2 = nn.Linear(fc_hidden, 1)

        # Policy head
        # Output spatial logits for each (from, to) combination
        # Uses a convolutional approach: output 64+8 channels (to_squares + knight_attacks)
        self.policy_conv = nn.Conv2d(hidden_channels, 72, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(72)
        # Final linear to produce policy logits
        # 72 channels × 64 squares = 4608 outputs
        self.policy_fc = nn.Linear(72 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning only value (for backward compatibility).

        Args:
            x: Input tensor of shape (batch, INPUT_CHANNELS, 8, 8)

        Returns:
            Value tensor of shape (batch, 1) in range [-1, 1]
        """
        value, _ = self.forward_policy_value(x)
        return value

    def forward_policy_value(
        self,
        x: torch.Tensor,
        legal_move_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy and value.

        Args:
            x: Input tensor of shape (batch, INPUT_CHANNELS, 8, 8)
            legal_move_mask: Optional bool tensor of shape (batch, POLICY_SIZE)
                            True for legal moves, False for illegal

        Returns:
            Tuple of:
            - value: Shape (batch, 1) in range [-1, 1]
            - policy: Shape (batch, POLICY_SIZE) - log probabilities
        """
        # Shared trunk
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)  # Flatten: (batch, 72*64)
        policy_logits = self.policy_fc(p)  # (batch, POLICY_SIZE)

        # Apply legal move mask if provided
        if legal_move_mask is not None:
            # Mask illegal moves with large negative value
            policy_logits = policy_logits.masked_fill(~legal_move_mask, -1e9)

        # Log softmax for numerical stability
        policy = F.log_softmax(policy_logits, dim=-1)

        return value, policy

    def get_policy_probs(
        self,
        x: torch.Tensor,
        legal_move_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get policy probabilities (not log probs)."""
        _, log_policy = self.forward_policy_value(x, legal_move_mask)
        return torch.exp(log_policy)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get network configuration for serialization."""
        return {
            "hidden_channels": self.hidden_channels,
            "num_blocks": self.num_blocks,
            "fc_hidden": self.fc_hidden,
            "network_type": "policy_value",
        }

    @classmethod
    def from_config(cls, config: dict) -> "PolicyValueNetwork":
        """Create network from config dict."""
        return cls(
            hidden_channels=config.get("hidden_channels", 64),
            num_blocks=config.get("num_blocks", 4),
            fc_hidden=config.get("fc_hidden", 128),
        )


def encode_move(move) -> int:
    """
    Encode a Move object to a policy index.

    Args:
        move: Move object with from_pos, to_pos, attack_pos, move_type

    Returns:
        Policy index in range [0, POLICY_SIZE)
    """
    from .game_simulator import MoveType

    from_sq = move.from_pos[0] + move.from_pos[1] * 8
    to_sq = move.to_pos[0] + move.to_pos[1] * 8

    if move.move_type == MoveType.MOVE_AND_ATTACK and move.attack_pos:
        # Knight combo: encode destination + attack direction
        # Base index for destination
        base_idx = from_sq * 64 + to_sq

        # Also encode attack direction (for MCTS move selection)
        # Attack direction relative to landing square
        ax, ay = move.attack_pos
        tx, ty = move.to_pos
        dx, dy = ax - tx, ay - ty

        # Cardinal directions: (0,1)=0, (0,-1)=1, (1,0)=2, (-1,0)=3
        # For simplicity, just use the from->to encoding
        return base_idx
    else:
        # MOVE, ATTACK, ABILITY: simple from*64 + to encoding
        return from_sq * 64 + to_sq


def decode_move_index(idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Decode a policy index back to (from_pos, to_pos).

    Note: This doesn't recover attack_pos for Knight combos.
    """
    if idx < POLICY_FROM_TO_SIZE:
        from_sq = idx // 64
        to_sq = idx % 64
        from_pos = (from_sq % 8, from_sq // 8)
        to_pos = (to_sq % 8, to_sq // 8)
        return from_pos, to_pos
    else:
        # Knight attack encoding - decode landing square and direction
        knight_idx = idx - POLICY_FROM_TO_SIZE
        landing_sq = knight_idx // 8
        direction = knight_idx % 8
        landing_pos = (landing_sq % 8, landing_sq // 8)
        return landing_pos, landing_pos  # Simplified


def create_legal_move_mask(moves: list, device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask tensor for legal moves.

    Args:
        moves: List of Move objects
        device: Torch device

    Returns:
        Boolean tensor of shape (POLICY_SIZE,) with True for legal moves
    """
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool, device=device)
    for move in moves:
        idx = encode_move(move)
        if 0 <= idx < POLICY_SIZE:
            mask[idx] = True
    return mask


class PersonalityValueNetwork(ExchangeValueNetwork):
    """
    Value network variant that supports personality-based evaluation.

    Adds an auxiliary output for personality traits that can be weighted
    during inference to shift evaluation towards different play styles:
    - Aggression: Bonus for attack opportunities
    - Defense: Bonus for piece safety
    - King focus: Bonus for king attack/safety
    - Material: Bonus for piece value differences
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        fc_hidden: int = 128,
        num_traits: int = 4,
    ):
        super().__init__(hidden_channels, num_blocks, fc_hidden)

        # Additional trait prediction head
        self.trait_fc = nn.Linear(fc_hidden, num_traits)
        self.num_traits = num_traits

        # Personality weights (can be adjusted at inference)
        self.register_buffer(
            "personality_weights",
            torch.zeros(num_traits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with personality weighting."""
        # Input processing
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Value head (shared features)
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        features = F.relu(self.value_fc1(v))

        # Base value
        base_value = torch.tanh(self.value_fc2(features))

        # Trait predictions (raw, not normalized)
        traits = self.trait_fc(features)

        # Apply personality weighting
        personality_bonus = (traits * self.personality_weights).sum(dim=1, keepdim=True)

        # Combine with clipping
        value = torch.tanh(base_value + 0.1 * personality_bonus)

        return value

    def forward_with_traits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both value and trait predictions."""
        out = F.relu(self.input_bn(self.input_conv(x)))

        for block in self.res_blocks:
            out = block(out)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        features = F.relu(self.value_fc1(v))

        value = torch.tanh(self.value_fc2(features))
        traits = self.trait_fc(features)

        return value, traits

    def set_personality(self, weights: list[float]) -> None:
        """Set personality weights for inference."""
        assert len(weights) == self.num_traits
        self.personality_weights.copy_(torch.tensor(weights))


def create_network(
    network_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function to create networks.

    Args:
        network_type: "standard", "policy_value", or "personality"
        **kwargs: Additional arguments for network constructor

    Returns:
        Configured network instance
    """
    if network_type == "standard":
        return ExchangeValueNetwork(**kwargs)
    elif network_type == "policy_value":
        return PolicyValueNetwork(**kwargs)
    elif network_type == "personality":
        return PersonalityValueNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# Model size presets
PRESETS = {
    "tiny": {"hidden_channels": 32, "num_blocks": 2, "fc_hidden": 64},   # ~30k params
    "small": {"hidden_channels": 48, "num_blocks": 3, "fc_hidden": 96},  # ~70k params
    "medium": {"hidden_channels": 64, "num_blocks": 4, "fc_hidden": 128},  # ~150k params
    "large": {"hidden_channels": 96, "num_blocks": 6, "fc_hidden": 192},  # ~400k params
}


def create_from_preset(preset: str = "medium", **overrides) -> ExchangeValueNetwork:
    """Create value-only network from size preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

    config = {**PRESETS[preset], **overrides}
    return ExchangeValueNetwork(**config)


def create_policy_value_from_preset(preset: str = "medium", **overrides) -> PolicyValueNetwork:
    """Create policy-value network from size preset (for MCTS training)."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

    config = {**PRESETS[preset], **overrides}
    return PolicyValueNetwork(**config)
