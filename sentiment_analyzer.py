import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RobertaSentimentAnalyzer:
    def __init__(self, model_name='roberta-base', num_labels=3):
        print(f"Initializing {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)

    def generate_sentiment_labels(self, text):
        """Generate sentiment labels using TextBlob"""
        try:
            if pd.isna(text) or not isinstance(text, str):
                return 1  # Neutral for invalid text
            
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            # Adjust for specific phrases
            lower_text = text.lower()
            
            # Strongly negative phrases
            if any(phrase in lower_text for phrase in ['terrible', 'worst', 'horrible', 'waste']):
                polarity -= 0.3
            
            # Strongly positive phrases
            if any(phrase in lower_text for phrase in ['excellent', 'perfect', 'amazing', 'love']):
                polarity += 0.3
            
            # Convert to classes
            if polarity < -0.1:
                return 0  # Negative
            elif polarity > 0.1:
                return 2  # Positive
            else:
                return 1  # Neutral
                
        except Exception as e:
            print(f"Error in sentiment generation: {e}")
            return 1  # Default to neutral

    def prepare_data(self, df, text_column='content'):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Convert content to string and clean
        df[text_column] = df[text_column].fillna('').astype(str)
        
        # Remove empty texts
        df = df[df[text_column].str.strip().str.len() > 0]
        
        # Generate sentiment labels
        print("Generating sentiment labels...")
        df['sentiment'] = df[text_column].apply(self.generate_sentiment_labels)
        
        # Split data
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42,
            stratify=df['sentiment']
        )
        
        print("\nSentiment Distribution:")
        print(df['sentiment'].value_counts())
        print("\nSentiment Distribution by Brand:")
        print(pd.crosstab(df['brand'], df['sentiment']))
        
        return train_df, val_df

    def save_model(self, path='./saved_model'):
        """Save the model and tokenizer"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save model configuration
            config = {
                'model_name': self.model_name,
                'num_labels': self.num_labels
            }
            torch.save(config, os.path.join(path, 'config.pt'))
            
            print(f"\nModel saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, path='./saved_model'):
        """Load a saved model"""
        try:
            # Load configuration
            config = torch.load(os.path.join(path, 'config.pt'))
            
            # Initialize with saved config
            analyzer = cls(
                model_name=config['model_name'],
                num_labels=config['num_labels']
            )
            
            # Load model and tokenizer
            analyzer.model = AutoModelForSequenceClassification.from_pretrained(path)
            analyzer.tokenizer = AutoTokenizer.from_pretrained(path)
            
            analyzer.model.to(analyzer.device)
            print(f"\nModel loaded successfully from {path}")
            
            return analyzer
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def train_model(
        self, 
        train_df, 
        val_df, 
        text_column='content',
        label_column='sentiment',
        batch_size=16,
        epochs=2,
        max_length=128,
        learning_rate=2e-5,
        save_path='./saved_model'
    ):
        """Train the model with optimized parameters for CPU"""
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_df[text_column].values,
            train_df[label_column].values,
            self.tokenizer,
            max_length
        )
        val_dataset = SentimentDataset(
            val_df[text_column].values,
            val_df[label_column].values,
            self.tokenizer,
            max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_accuracy = 0
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_progress = tqdm(train_loader, desc='Training')
            for batch in train_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                train_progress.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'acc': f'{100 * train_correct/train_total:.2f}%'
                })

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = 100 * val_correct / val_total
            
            print(f'\nTraining Loss: {avg_train_loss:.3f}')
            print(f'Training Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Accuracy: {val_accuracy:.2f}%')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # Save best model
                self.save_model(save_path)

        return best_accuracy

    def predict(self, texts, batch_size=16):
        """Predict sentiment for new texts"""
        self.model.eval()
        predictions = []
        
        dataset = SentimentDataset(
            texts,
            [0] * len(texts),
            self.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions

# Example usage
if __name__ == "__main__":
    try:
        # Check if model exists
        if os.path.exists('./saved_model'):
            print("Loading existing model...")
            analyzer = RobertaSentimentAnalyzer.load_model('./saved_model')
        else:
            print("Training new model...")
            # Load data
            df = pd.read_csv('all_brand_reviews.csv')
            
            # Drop unnecessary columns
            columns_to_drop = ['title', 'rating']
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            if existing_columns:
                df = df.drop(columns=existing_columns)
            
            # Initialize analyzer
            analyzer = RobertaSentimentAnalyzer('roberta-base')
            
            # Prepare data
            train_df, val_df = analyzer.prepare_data(df)
            
            # Train model
            best_accuracy = analyzer.train_model(
                train_df,
                val_df,
                batch_size=16,
                epochs=2,
                max_length=128,
                save_path='./saved_model'
            )
            
            print(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
        
        # Example predictions
        test_texts = [
            "These shoes are perfect for long distance running. Very comfortable.",
            "The shoes started falling apart after just two months.",
            "They are okay for the price, but nothing special."
        ]
        
        predictions = analyzer.predict(test_texts)
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        print("\nExample Predictions:")
        for text, pred in zip(test_texts, predictions):
            print(f"\nText: {text}")
            print(f"Sentiment: {sentiment_map[pred]}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise