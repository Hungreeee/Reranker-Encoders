import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from encoders import BiEncoder, CrossEncoder, PolyEncoder
from transformers import AutoModel, AutoTokenizer

from utils import *
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, type=str, help="The name of the training run")

    parser.add_argument("--encoder", required=True, type=str, help="The encoder model to be trained")
    parser.add_argument("--poly_m", default=16, type=int, help="Poly encoder's number of global features")

    parser.add_argument("--base_transformer", default="prajjwal1/bert-tiny", type=str, help="The base BERT transformer.")
    parser.add_argument("--dataset", default="ms_marco", type=str)
    parser.add_argument("--dataset_configs", default="./dataset_configs.cfg", type=str)
    parser.add_argument("--output_dir", default="./model_checkpoints/", type=str)

    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    
    parser.add_argument("--num_epoch", default=15, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=int)
    parser.add_argument("--weight_decay", default=1e-2, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--save_steps", default=500, type=int)

    parser.add_argument("--log_results", default=True, type=bool, help="Whether to log the validation results during training.")
    parser.add_argument('--device', default="cpu", type=str)

    args = parser.parse_args()
    log_writer = open(os.path.join(args.output_dir, 'logs.txt'), 'a', encoding='utf-8')

    base_model = AutoModel.from_pretrained(args.base_transformer)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_transformer)

    print("Base transformer loaded.")

    dataset_configs = read_config_file(args.dataset_configs)
    trainset, valset = load_huggingface_dataset(args.dataset, dataset_configs["DATASET"], base_tokenizer, args.max_seq_length)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    print("Train and validation set loaded.")

    if args.encoder == "bi":
        model = BiEncoder(transformer=base_model, device=args.device)
    elif args.encoder == "cross":
        model = CrossEncoder(transformer=base_model, device=args.device)
    elif args.encoder == "poly":
        model = PolyEncoder(transformer=base_model, num_global_features=args.poly_m, device=args.device)

    model = model.to(args.device)
    print("Encoder model loaded.")

    print(f"Number of trainable parameters: {count_parameters(model.parameters())}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    criterion = torch.nn.CrossEntropyLoss()

    print("Training started...")
    train_loss, val_loss = train(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epoch=args.num_epoch,
        device=args.device,
        eval=True
    )

    log_writer.write(f"Run name: {args.run_name}\n")
    log_writer.write(''.join([f'Epoch {i + 1}/{args.num_epoch}: Train loss = {loss[0]} | Validation loss = {loss[1]}\n' for i, loss in enumerate(zip(train_loss, val_loss))]) + '\n')

    model_save_path = args.output_dir / ("_".join([args.run_name, args.encoder]) + ".pth")
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}.")

