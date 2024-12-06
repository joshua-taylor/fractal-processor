import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
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
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
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
    def __init__(self, d_model, num_patterns=128):
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

        # **Pattern-based representations**
        # Score each token's relationship to each pattern
        scores = self.scorer(x)  # [B, L, num_patterns]
        if causal_mask is None:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask[:L, :L].unsqueeze(-1), float('-inf'))
        scores = F.softmax(scores, dim=-1)

        # Create pattern-based representations
        pattern_mix = torch.einsum('bln,nd->bld', scores, self.patterns)

        # **Combine with Tokens**
        # Combine tokens with pattern-based connections
        combined = torch.cat([x, pattern_mix], dim=-1)
        combined = self.mixer(combined)

        return combined + x  # Add residual


class CausalFractalProcessor(nn.Module):
    def __init__(self, vocab_size, d_model, depth=3, window_size=32, num_convolutions=3,
                        use_embedding=True,
                        use_convolutions=True,
                        use_token_connectors=True,
                        use_predictors=True,
                        use_scale_mixer=True,
                        use_pos_phases=True,):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.window_size = window_size

        # Store the toggles
        self.use_embedding = use_embedding
        self.use_convolutions = use_convolutions
        self.use_token_connectors = use_token_connectors
        self.use_predictors = use_predictors
        self.use_scale_mixer = use_scale_mixer
        self.use_pos_phases = use_pos_phases
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Progressive window sizes and dilations
        self.windows = [window_size * (2**i) for i in range(depth)]
        self.dilations = [i+1 for i in range(depth)]
        
        # Causal convolutions
        self.causal_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    d_model, 
                    d_model, 
                    kernel_size=max(7, window),
                    padding=0,  # No padding to maintain causality
                    dilation=self.dilations[i], 
                    groups=d_model
                ) for _ in range(num_convolutions)
            ]) for i, window in enumerate(self.windows)
        ])
        
        # Token connectors
        self.token_connectors = nn.ModuleList([
            TokenConnector(d_model) for _ in range(depth)
        ])
        
        # Predictors
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(depth)
        ])
        
        # Scale mixer
        self.scale_mixer = nn.ModuleList([
            nn.Linear(d_model * (i + 1), d_model)
            for i in range(depth)
        ])
        
        # Positional phases
        self.pos_phases = nn.Parameter(torch.randn(depth, d_model) * 0.02)
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(depth)
        ])
        
        # Output projection to vocabulary
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, vocab_size)
        )

    def _apply_causal_padding(self, x, conv):
        pad_size = (conv.kernel_size[0] - 1) * conv.dilation[0]
        return F.pad(x, (pad_size, 0))

    def forward(self, x, use_embedding=True, use_convolutions=True, 
                use_token_connectors=True, use_predictors=True, 
                use_scale_mixer=True, use_pos_phases=True):
        B, L = x.shape

        # Embedding or one-hot representation
        if use_embedding:
            x = self.embedding(x)  # [B, L, d_model]
        else:
            x = F.one_hot(x, num_classes=self.d_model).float()

        outputs = []
        current_features = x
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        for i in range(self.depth):
            scale_input = current_features

            # Add positional information if enabled
            if use_pos_phases:
                pos = torch.arange(L, device=x.device)[:, None] * self.pos_phases[i][None, :]
                scale_input = scale_input * (1 + pos.sin()) + pos.cos()

            # Layer normalization
            scale_input = self.layer_norms[i](scale_input)

            # Apply causal convolutions
            if use_convolutions:
                conv_input = scale_input.transpose(1, 2)  # [B, d_model, L]
                conv_input = self._apply_causal_padding(conv_input, self.causal_convs[i][0])

                conv_outputs = [conv(conv_input) for conv in self.causal_convs[i]]
                conv_out = torch.stack(conv_outputs, dim=2)  # [B, d_model, L, num_convs]
                conv_out, _ = torch.max(conv_out, dim=2)  # [B, d_model, L]
                conv_out = conv_out.transpose(1, 2)  # [B, L, d_model]
            else:
                conv_out = scale_input

            # Apply token connectors
            if use_token_connectors:
                connected_out = self.token_connectors[i](conv_out, causal_mask)
            else:
                connected_out = conv_out

            # Predict future context
            if use_predictors:
                pred_out = self.predictors[i](connected_out)
                scale_out = connected_out + 0.1 * pred_out + scale_input
            else:
                scale_out = connected_out + scale_input

            # Scale mixing
            if use_scale_mixer and outputs:
                prev_scales = torch.cat([*outputs, scale_out], dim=-1)
                scale_out = self.scale_mixer[i](prev_scales)

            outputs.append(scale_out)
            current_features = scale_out

        # Final output projection
        logits = self.output_layer(outputs[-1])  # [B, L, vocab_size]
        return logits





class WikiSequentialDataset(Dataset):
    def __init__(self, tokenizer, split="train", sequence_length=128,max_rows=None,build_tokenizer=True):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.max_rows = max_rows
        self.dataset_vocab = Counter()  # To count token occurrences
        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        # Limit the dataset to the first max_rows if specified
        if max_rows:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        # Process text in chunks
        self.sequences = []
        chunk_size = 100000
        # text cleaning specific to the wikitext dataset but should not really impact other datasets:
        full_text = " ".join([text.strip() for text in dataset["text"] if text.strip()])
        # Step 1: Replace " @" with ""
        full_text = full_text.replace(" @", "")
        
        # Step 2: Replace "@ " with ""
        full_text = full_text.replace("@ ", "")
        
        # Step 3: Remove all remaining "@" symbols
        full_text = full_text.replace("@", "")
        
        if build_tokenizer:
            # Build a tokenizer
            tokenizer = tokenizer.train_new_from_iterator([full_text], 10000)
            tokenizer.save_pretrained("custom-tokenizer")
            self.tokenizer = tokenizer
        
        # Build dataset_vocab and prepare sequences
        self.dataset_vocab = Counter()
        self.sequences = []

        # Tokenize and build vocabulary
        chunk_size = 100000
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            tokens = tokenizer.encode(chunk)  # Get the token IDs as a list
            self.dataset_vocab.update(tokens)

            for j in range(0, len(tokens) - sequence_length - 1, sequence_length):
                input_sequence = tokens[j:j + sequence_length]
                target_sequence = tokens[j + 1:j + sequence_length + 1]
                if len(input_sequence) == sequence_length and len(target_sequence) == sequence_length:
                    self.sequences.append((input_sequence, target_sequence))

        print(f"Created {len(self.sequences)} sequences of length {sequence_length}")
        print(f"Vocabulary in dataset: {len(self.dataset_vocab)}")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids, target_ids = self.sequences[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }




def generate_sample(
    model, tokenizer, input_ids, device, max_new_tokens=50, beam_width=2,
    temperature=0.7, diverse_penalty=0.5, contrastive_penalty=0.1, max_input_length=128
):
    model.eval()
    with torch.no_grad():
        # Initialize beams
        beams = [(input_ids.clone().to(device), 0)]  # List of (sequence, score)
        
        for _ in range(max_new_tokens):
            new_beams = []
            
            # Expand each beam
            for seq, score in beams:
                # Ensure the input sequence adheres to the max_input_length
                if seq.shape[1] > max_input_length:
                    seq = seq[:, -max_input_length:]  # Keep the last `max_input_length` tokens
                
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
        input_text = tokenizer.decode(batch["input_ids"][i].tolist(), skip_special_tokens=True)
        # Generate continuation
        continuation = generate_sample(
            model, 
            tokenizer, 
            batch["input_ids"][i:i+1].to(device),
            device
        )
        
        # Get predictions and targets
        predictions = torch.argmax(logits[i], dim=-1)
        # Squeeze the predictions tensor to remove batch dimension if it's size 1
        predictions = predictions.squeeze(0)  # Now predictions should have shape (seq_length,)
        targets = batch["labels"][i]
        
        
        # Decode predictions
        pred_text = tokenizer.decode(predictions.tolist(), skip_special_tokens=True)
        target_text = tokenizer.decode(targets.tolist(), skip_special_tokens=True)
        
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
    

    def compute_loss(
        self, logits, labels, label_smoothing=0.1, repetition_penalty=1.2, coverage_weight=0.5
    ):
        """
        Compute loss for language modeling, including cross-entropy, repetition, and coverage penalties.
    
        Args:
            logits: Predicted logits from the model (batch_size, seq_len, vocab_size).
            labels: Ground truth token IDs (batch_size, seq_len).
            label_smoothing: Smoothing for cross-entropy loss (default=0.1).
            repetition_penalty: Penalize repeated predictions (default=1.2).
            coverage_weight: Weight for encouraging diverse token coverage (default=0.5).
    
        Returns:
            Combined loss (scalar).
        """
        batch_size, sequence_length, vocab_size = logits.shape
    
        # Reshape logits and labels for loss computation
        logits = logits.view(-1, vocab_size)  # Shape: [batch_size*seq_len, vocab_size]
        labels = labels.view(-1)  # Shape: [batch_size*seq_len]
    
        # Cross-entropy loss with label smoothing
        ce_loss = F.cross_entropy(
            logits, labels, ignore_index=-100, label_smoothing=label_smoothing
        )
    
        # Repetition penalty - discourage repeated predictions in logits
        probs = F.softmax(logits, dim=-1)  # Shape: [batch_size*seq_len, vocab_size]
        top_probs, _ = probs.topk(1, dim=-1)  # Top predicted probabilities, shape: [batch_size*seq_len, 1]
        repeated_penalty = (top_probs ** 2).mean()  # Penalize highly repeated predictions
        repetition_loss = repetition_penalty * repeated_penalty
    
        # Coverage loss - encourage predictions to align with true labels
        one_hot_labels = F.one_hot(labels, num_classes=vocab_size).float()  # Shape: [batch_size*seq_len, vocab_size]
        coverage_loss = coverage_weight * torch.sum(probs * one_hot_labels) / labels.size(0)
    
        # Combine losses
        combined_loss = ce_loss + repetition_loss - coverage_loss  # Coverage as reward (-ve loss)
    
        return combined_loss


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
        print(f'tokenizer size = {tokenizer.vocab_size}')
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
                if self.step > 0 and self.step % self.config["eval_steps"] == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": self.step})
                    self.save_checkpoint(val_loss)
                    self.model.train()
                
                # Generate samples
                if self.step > 0 and self.step % self.config["sample_steps"] == 0:
                    log_samples(self.model, batch, self.tokenizer, self.device, self.step)
                    self.model.train()
                self.step += 1
                pbar.update(1)
                
                if self.step >= self.config["total_steps"]:
                    break
            
            pbar.set_description(f"Training Epoch {epoch}")

def train_model(model_class, model_args,model=None):
    # Initialize wandb
    run = wandb.init(project="neural-wave-models", name=f"{model_class.__name__}_experiment")

    # Training configuration
    config = {
        "batch_size": 32,
        "learning_rate": 1.5e-3,
        "max_length": 128,
        "total_steps": 40000,
        "eval_steps": 500,
        "sample_steps": 100,
        "warmup_steps": 200,
        "max_rows": 300000, ##limit data size
        **model_args
    }
    wandb.config.update(config)
    
    # Setup
    save_dir = Path(f"checkpoints/{run.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # legacy - set as none as not impacting model
    if not model: # if no model then build a tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = Tokenizer.from_file("custom_tokenizer/tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        special_tokens_dict = {
            'unk_token': '<|endoftext|>',
            'bos_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>'
        }
        tokenizer.add_special_tokens(special_tokens_dict)
    
    # Create datasets
    if not model: # if no model then build a tokenizer
        train_dataset = WikiSequentialDataset(tokenizer, split="train", sequence_length=config["max_length"],
                                        max_rows=config.get("max_rows", None))
        tokenizer = train_dataset.tokenizer
    else:
        train_dataset = WikiSequentialDataset(tokenizer, split="train", sequence_length=config["max_length"],
                                        max_rows=config.get("max_rows", None),build_tokenizer=False)
    
    val_dataset = WikiSequentialDataset(tokenizer, split="validation", 
                                      sequence_length=config["max_length"],build_tokenizer=False)
    

    
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
    if not model:
        model = model_class(**model_args, vocab_size=len(tokenizer.get_vocab())).to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=config["total_steps"],
        pct_start=config["warmup_steps"] / config["total_steps"],
        div_factor=10,                           # Default: 25; reduces starting LR
        final_div_factor=0.2                       # Default: 1e4; makes final LR less aggressive
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
    
    return model, optimizer, scheduler, checkpoint, tokenizer


#### TO TRAIN! ####

results = train_model(CausalFractalProcessor, fractal_model_args) #,model=model #add model to continue training...
