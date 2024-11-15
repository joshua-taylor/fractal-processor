import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import json
from pathlib import Path
from tqdm.auto import tqdm
import math
import random
from datetime import datetime
from collections import Counter
import itertools
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, RepetitionPenaltyLogitsProcessor

class LSTMProcessor(nn.Module):
    def __init__(self, vocab_size, d_model, depth=3, window_size=32):
        """
        A minimal LSTM-based language model for benchmarking.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the model (embedding & hidden state size)
            depth (int): Number of LSTM layers
            window_size (int): Not used, kept for compatibility with fractal model
        """
        super().__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Single LSTM with specified number of layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=depth,
            batch_first=True
        )
        
        # Simple linear output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # Get embeddings
        x = self.embedding(x)  # [batch_size, sequence_length, d_model]
        
        # Process through LSTM
        x, _ = self.lstm(x)  # Ignore hidden state output
        
        # Project to vocabulary size
        logits = self.output_layer(x)
        
        return logits  # shape: [batch_size, sequence_length, vocab_size]


class TokenConnector(nn.Module):
    def __init__(self, d_model, num_patterns=4):
        super().__init__()
        self.d_model = d_model
        
        # Simple learned patterns for token relationships
        self.patterns = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)
        
        # Lightweight scorer to identify important tokens
        self.scorer = nn.Linear(d_model, num_patterns)
        
        # Simple mixing layer
        self.mixer = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x, causal_mask=None):
        B, L, D = x.shape
        
        # Score each token's relationship to each pattern
        scores = self.scorer(x)  # [B, L, num_patterns]
        
        # Create causal mask for this sequence length if provided
        if causal_mask is None:
            # Create mask of appropriate size
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask[:L, :L].unsqueeze(-1), float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        
        # Create pattern-based representations
        pattern_mix = torch.einsum('bln,nd->bld', scores, self.patterns)
        
        # Combine original tokens with their pattern-based connections
        output = self.mixer(torch.cat([x, pattern_mix], dim=-1))
        
        return output + x  # Add residual connection

class CausalFractalProcessor(nn.Module):
    def __init__(self, vocab_size, d_model, depth=3, window_size=32):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.window_size = window_size
        
        # Add embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Progressive window sizes for each level
        self.windows = [window_size * (2**i) for i in range(depth)]
        
        # Causal convolutions for each scale
        self.causal_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, 
                d_model, 
                kernel_size=max(7, window),
                padding=0,  # No padding - maintain causality
                groups=d_model  # Depthwise for efficiency
            ) for window in self.windows
        ])
        
        # Add token connectors for each scale
        self.token_connectors = nn.ModuleList([
            TokenConnector(d_model) for _ in range(depth)
        ])
        
        # Future-prediction modules
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(depth)
        ])
        
        # Scale mixing with causal constraint
        self.scale_mixer = nn.ModuleList([
            nn.Linear(d_model * (i + 1), d_model)
            for i in range(depth)
        ])
        
        # Learned positional phases
        self.pos_phases = nn.Parameter(torch.randn(depth, d_model) * 0.02)
        
        # Layer normalization for better stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(depth)
        ])
        
        # Output projection to vocabulary
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, vocab_size)
        )

    def _apply_causal_padding(self, x, conv):
        # Apply causal padding based on kernel size
        pad_size = conv.kernel_size[0] - 1
        return F.pad(x, (pad_size, 0))

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        B, L = x.shape
        
        # Convert tokens to embeddings
        x = self.embedding(x)  # shape: [batch_size, sequence_length, d_model]
        
        outputs = []
        current_features = x
        
        # Create causal mask once for all token connectors
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        
        # Process each scale causally
        for i in range(self.depth):
            # Add positional information using phases
            pos = torch.arange(L, device=x.device)[:, None] * self.pos_phases[i][None, :]
            scale_input = current_features * (1 + pos.sin()) + pos.cos()
            
            # Apply layer normalization
            scale_input = self.layer_norms[i](scale_input)
            
            # Prepare for causal convolution
            conv_input = scale_input.transpose(1, 2)  # [batch_size, d_model, sequence_length]
            conv_input = self._apply_causal_padding(conv_input, self.causal_convs[i])
            
            # Apply causal convolution
            conv_out = self.causal_convs[i](conv_input)
            conv_out = conv_out.transpose(1, 2)  # [batch_size, sequence_length, d_model]
            
            # Apply token connection with causal masking
            connected_out = self.token_connectors[i](conv_out, causal_mask)
            
            # Predict future context
            pred_out = self.predictors[i](connected_out)
            
            # Combine current and predicted features with residual connection
            scale_out = connected_out + 0.1 * pred_out + scale_input
            
            # Progressive mixing of scales
            if outputs:
                prev_scales = torch.cat([*outputs, scale_out], dim=-1)
                scale_out = self.scale_mixer[i](prev_scales)
            
            outputs.append(scale_out)
            current_features = scale_out
        
        # Final output projection to vocabulary size
        logits = self.output_layer(outputs[-1])
        
        return logits  # shape: [batch_size, sequence_length, vocab_size]




class WikiSequentialDataset(Dataset):
    def __init__(self, tokenizer, split="train", sequence_length=128,test_size=None):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.dataset_vocab = Counter()  # To count token occurrences
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Process text in chunks
        self.sequences = []
        chunk_size = 100000
        
        full_text = " ".join([text.strip() for text in dataset["text"] if text.strip()])
        # Step 1: Replace " @" with ""
        full_text = full_text.replace(" @", "")
        
        # Step 2: Replace "@ " with ""
        full_text = full_text.replace("@ ", "")
        
        # Step 3: Remove all remaining "@" symbols
        full_text = full_text.replace("@", "")
        
        if test_size:
            full_text = full_text[0:test_size]
        # Tokenize and build vocabulary
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            tokens = self.tokenizer(chunk, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            self.dataset_vocab.update(tokens.tolist())
            
            for j in range(0, len(tokens) - sequence_length - 1, sequence_length):
                input_sequence = tokens[j:j + sequence_length]
                target_sequence = tokens[j + 1:j + sequence_length + 1]
                if len(input_sequence) == sequence_length and len(target_sequence) == sequence_length:
                    self.sequences.append((input_sequence, target_sequence))
        
        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        print(f"Vocabulary reduced to {len(self.dataset_vocab)} tokens from original {len(tokenizer)}")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, target_ids = self.sequences[idx]
        return {
            "input_ids": input_ids,
            "labels": target_ids
        }

def reduce_tokenizer_vocab(tokenizer, dataset_vocab):
    # Filter out tokens not in the dataset vocabulary
    filtered_vocab = {k: v for k, v in tokenizer.get_vocab().items() if v in dataset_vocab}
    # Create a new directory to save the reduced vocabulary and config
    vocab_dir = Path("reduced_tokenizer")
    vocab_dir.mkdir(exist_ok=True)
    # Save the filtered vocabulary as a JSON file
    vocab_file_path = vocab_dir / "vocab.json"
    with open(vocab_file_path, "w") as vocab_file:
        json.dump(filtered_vocab, vocab_file)
    # Copy the tokenizer config files to the new directory
    tokenizer.save_pretrained(vocab_dir)

    # Load the tokenizer from the newly created directory
    reduced_tokenizer = AutoTokenizer.from_pretrained(vocab_dir,
        vocab_size=len(filtered_vocab))
    print(f'size of tokenizer vocab is now: {len(reduced_tokenizer)}')
    return reduced_tokenizer


def generate_sample(model, tokenizer, input_ids, device, max_new_tokens=50, beam_width=5, temperature=0.7, diverse_penalty=0.5, contrastive_penalty=0.1):
    model.eval()
    with torch.no_grad():
        # Initialize beams
        beams = [(input_ids.clone().to(device), 0)]  # List of (sequence, score)
        
        for _ in range(max_new_tokens):
            new_beams = []
            
            # Expand each beam
            for seq, score in beams:
                # Get logits for the next token
                outputs = model(seq)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature
                
                # Convert logits to probabilities
                probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, beam_width, dim=-1)
                
                # Expand each sequence in the beam with each of the top-k tokens
                for i in range(beam_width):
                    next_token = top_k_indices[:, i]
                    next_token_prob = top_k_probs[:, i]
                    
                    # Contrastive penalty: discourage tokens that repeat recent tokens in the sequence
                    if next_token in seq[0, -5:]:  # Check last 5 tokens for repeats
                        penalty = contrastive_penalty * (seq[0, -5:] == next_token).sum().item()
                        next_token_prob -= penalty
                    
                    # Update sequence and score
                    new_seq = torch.cat([seq, next_token.unsqueeze(-1)], dim=1)
                    new_score = score + next_token_prob.item()
                    
                    # Optional: Apply diversity penalty
                    if diverse_penalty > 0:
                        new_score -= diverse_penalty * len([1 for tok in new_seq[0] if tok == next_token])
                    
                    # Add to new beams
                    new_beams.append((new_seq, new_score))
            
            # Select top beams by score
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check for end-of-sequence token
            if all(tokenizer.eos_token_id in beam[0] for beam in beams):
                break
        
        # Return the best beam
        best_beam = max(beams, key=lambda x: x[1])[0]
        return tokenizer.decode(best_beam[0], skip_special_tokens=True)

def log_samples(model, batch, tokenizer, device, step):
    samples = []
    
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device))
    
    for i in range(min(3, len(batch["input_ids"]))):
        # Get input text
        input_text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
        
        # Generate continuation
        continuation = generate_sample(
            model, 
            tokenizer, 
            batch["input_ids"][i:i+1].to(device),
            device
        )
        
        # Get predictions and targets
        predictions = torch.argmax(logits[i], dim=-1)
        targets = batch["labels"][i]
        
        # Collect the full predictions and corresponding targets
        pred_text = tokenizer.decode(predictions, skip_special_tokens=True)
        target_text = tokenizer.decode(targets, skip_special_tokens=True)
        
        samples.append({
            "step": step,
            "input": input_text,
            "continuation": continuation[120:],
            "predictions": pred_text,
            "targets": target_text
        })
    
    # Log to wandb
    wandb.log({
        "samples": wandb.Table(
            columns=["step", "input", "continuation", "predictions", "targets"],
            data=[[s["step"], s["input"], s["continuation"], s["predictions"], s["targets"]] for s in samples]
        )
    })

    # Print out to console as well
    for sample in samples:
        print(f"Step: {sample['step']}")
        print(f"Targets: {sample['targets']}")
        print("\n" + "-"*50 + "\n")
        print(f"Continuation: {sample['continuation'][120:]}")
        print("\n" + "-"*50 + "\n")
        print(f"Predictions: {sample['predictions']}")
        print("\n" + "="*70 + "\n")
        print("="*70 + "\n")


class Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, 
                 optimizer, scheduler, device, save_dir, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.config = config
        
        self.step = 0
        self.best_val_loss = float('inf')
        self.last_save_time = datetime.now()
    
    def compute_loss(self, logits, labels, reduction_penalty=0.1):
        # Shape: [batch_size, sequence_length, vocab_size]
        batch_size, sequence_length, vocab_size = logits.shape
        
        # Compute cross entropy loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction='none'
        ).view(batch_size, sequence_length)
        
        # Apply exponential weight decay across sequence
        pos_weights = torch.exp(torch.linspace(0, reduction_penalty, sequence_length)).to(logits.device)
        weighted_loss = loss * pos_weights
        
        return weighted_loss.mean()
    
    def validate(self):
        print("Starting validation...")
        self.model.eval()
        total_loss = 0
        num_batches = min(len(self.val_dataloader), 50)
        
        with torch.no_grad():
            for i, batch in enumerate(itertools.islice(self.val_dataloader, num_batches)):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.compute_loss(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Validation complete. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, val_loss):
        try:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'step': self.step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }
                
                path = self.save_dir / f"best_model.pt"
                torch.save(checkpoint, path)
                wandb.save(str(path))
                print(f"\nSaved best model at step {self.step} with val_loss {val_loss:.4f}")
        except OSError as e:
            print(f"Warning: Could not save checkpoint due to OSError: {e}")

    def train(self):
        self.model.train()
        epoch = 0
        pbar = tqdm(total=self.config["total_steps"], desc=f"Training Epoch {epoch}")
        print("Starting training...")
        print(f"Total steps: {self.config['total_steps']}")
        print(f"Number of training batches: {len(self.train_dataloader)}")        
        while self.step < self.config["total_steps"]:
            epoch += 1
            print(f"Starting epoch {epoch}")
            
            for batch in self.train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                loss = self.compute_loss(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Logging
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "step": self.step
                })
                
                # Validation and checkpointing
                if self.step % self.config["eval_steps"] == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": self.step})
                    self.save_checkpoint(val_loss)
                    self.model.train()
                
                # Generate samples
                if self.step % self.config["sample_steps"] == 0:
                    log_samples(self.model, batch, self.tokenizer, self.device, self.step)
                    self.model.train()
                
                self.step += 1
                pbar.update(1)
                
                if self.step >= self.config["total_steps"]:
                    break
            
            pbar.set_description(f"Training Epoch {epoch}")

def train_model(model_class, model_args):
    # Initialize wandb
    run = wandb.init(project="neural-wave-models", name=f"{model_class.__name__}_experiment")

    # Training configuration
    config = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_length": 128,
        "total_steps": 10000,
        "eval_steps": 500,
        "sample_steps": 100,
        "warmup_steps": 200,
        #"test_size": 100000, ##token size
        **model_args
    }
    wandb.config.update(config)
    
    # Setup
    save_dir = Path(f"checkpoints/{run.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")
    
    # Create datasets
    val_dataset = WikiSequentialDataset(tokenizer, split="validation", 
                                      sequence_length=config["max_length"])
    train_dataset = WikiSequentialDataset(tokenizer, split="train", 
                                        test_size=config.get("test_size", None))
    # Reduce tokenizer vocabulary
    tokenizer = reduce_tokenizer_vocab(tokenizer, train_dataset.dataset_vocab)

    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # Initialize model
    model = model_class(**model_args, vocab_size=len(tokenizer)).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=config["total_steps"],
        pct_start=config["warmup_steps"] / config["total_steps"]
    )
    
    # Create trainer and start training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        config=config
    )
    
    trainer.train()
    wandb.finish()
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'device': device,
    }

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load a saved checkpoint and restore model, optimizer, and scheduler states.
    
    Args:
        path: Path to the checkpoint file
        model: The model architecture to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load the model to ('cuda' or 'cpu')
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer (if provided)
        scheduler: Loaded scheduler (if provided)
        checkpoint: Dictionary containing all loaded data
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Set model to evaluation mode by default
    model.eval()
    
    return model, optimizer, scheduler, checkpoint

####### FOR RUNNING THE BENCHMARK #########
# For the Causal Fractal model
fractal_model_args = {
    "d_model": 256,
    "depth": 4,
    "window_size": 8
}
fractal_results = train_model(CausalFractalProcessor, fractal_model_args)

model_args = {
    "d_model": 256,  # Embedding dimension
    "depth": 3      # Number of LSTM layers
}

results = train_model(LSTMProcessor, model_args)
