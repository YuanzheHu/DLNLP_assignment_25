import argparse
import os
from src.train import train_model
from src.evaluate import compare_models
from src.models.dual_input_hybrid import build_dual_input_hybrid_model
from src.models.token_bert import build_token_bert_model
from src.models.tribrid_use import build_tribrid_use_model
from src.models.tribrid_biobert import build_tribrid_biobert_model
from src.data import load_pubmed_rct_dataset

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on PubMed RCT dataset.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'train_and_evaluate'],
                      help='Mode to run: train, evaluate, or train_and_evaluate')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['dual_input_hybrid', 'token_bert', 'tribrid_use', 'tribrid_biobert', 'all'],
                      help='Model architecture to train/evaluate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to PubMed RCT data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint_path', type=str, default='best_weights/checkpoint.ckpt', 
                      help='Path to save/load model weights')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load label information
    _, _, _, label_info = load_pubmed_rct_dataset(args.data_dir)
    num_classes = label_info['num_classes']

    if args.mode in ['train', 'train_and_evaluate']:
        if args.model == 'all':
            models_to_train = ['dual_input_hybrid', 'token_bert', 'tribrid_use', 'tribrid_biobert']
        else:
            models_to_train = [args.model]

        for model_name in models_to_train:
            print(f"\nTraining {model_name} model...")
            checkpoint_path = os.path.join(args.checkpoint_path, f"{model_name}.ckpt")
            train_model(
                model_name=model_name,
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint_path,
                output_dir=args.output_dir
            )

    if args.mode in ['evaluate', 'train_and_evaluate']:
        if args.model == 'all':
            # Load all models
            models = {
                'dual_input_hybrid': build_dual_input_hybrid_model(num_classes=num_classes, output_seq_char_len=290, num_char_tokens=69),
                'token_bert': build_token_bert_model(num_classes=num_classes),
                'tribrid_use': build_tribrid_use_model(num_classes=num_classes),
                'tribrid_biobert': build_tribrid_biobert_model(num_classes=num_classes)
            }
            
            # Load weights for each model
            for model_name, model in models.items():
                checkpoint_path = os.path.join(args.checkpoint_path, f"{model_name}.ckpt")
                model.load_weights(checkpoint_path)
            
            # Compare all models
            print("\nComparing all models...")
            compare_models(
                models=models,
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )
        else:
            # Load single model
            if args.model == 'dual_input_hybrid':
                model = build_dual_input_hybrid_model(num_classes=num_classes, output_seq_char_len=290, num_char_tokens=69)
            elif args.model == 'token_bert':
                model = build_token_bert_model(num_classes=num_classes)
            elif args.model == 'tribrid_use':
                model = build_tribrid_use_model(num_classes=num_classes)
            elif args.model == 'tribrid_biobert':
                model = build_tribrid_biobert_model(num_classes=num_classes)
            
            # Load weights
            model.load_weights(args.checkpoint_path)
            
            # Evaluate single model
            print(f"\nEvaluating {args.model} model...")
            compare_models(
                models={args.model: model},
                data_dir=args.data_dir,
                output_dir=args.output_dir
            )

if __name__ == "__main__":
    main() 